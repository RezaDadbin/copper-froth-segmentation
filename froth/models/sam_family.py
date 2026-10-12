"""Gradient-preserving SAM/HQ-SAM preprocessing, prompts and proposal inference."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def prepare_image(model, image):
    """Resize longest side, normalize, then pad in SAM's input coordinate frame."""
    height, width = image.shape[:2]
    size = model.image_encoder.img_size
    scale = size / max(height, width)
    input_hw = (int(height * scale + 0.5), int(width * scale + 0.5))
    # SAM's official PIL resize/antialias implementation, selected lazily.
    try:
        from segment_anything.utils.transforms import ResizeLongestSide
    except ImportError:
        from segment_anything_hq.utils.transforms import ResizeLongestSide
    image = ResizeLongestSide(size).apply_image(image)
    tensor = torch.as_tensor(np.ascontiguousarray(image), device=model.device).permute(2, 0, 1).float()
    return model.preprocess(tensor).unsqueeze(0), input_hw


def box_prompt(mask, source):
    height, width = mask.shape
    if source == "full_image":
        return np.array([0, 0, width - 1, height - 1], np.float32)
    if source != "ground_truth":
        raise ValueError("Prompt source must be explicitly full_image or ground_truth")
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("Ground-truth box prompting is undefined for an empty annotation mask")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)


def encode_image(model, tensor, family):
    encoded = model.image_encoder(tensor)
    if family == "hq_sam":
        if not isinstance(encoded, tuple) or len(encoded) != 2:
            raise RuntimeError("HQ-SAM must return image embeddings and intermediate embeddings; check its package version.")
        return encoded
    if not torch.is_tensor(encoded):
        raise TypeError("SAM/MedSAM image encoder returned an unexpected object")
    return encoded, None


def decode(model, embedding, intermediate, family, *, boxes=None, points=None):
    sparse, dense = model.prompt_encoder(points=points, boxes=boxes, masks=None)
    kwargs = dict(image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
                  sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                  multimask_output=False)
    if family == "hq_sam":
        kwargs.update(hq_token_only=True, interm_embeddings=intermediate)
    return model.mask_decoder(**kwargs)


def prompted_logits(model, image, mask, family, prompt_source):
    tensor, input_hw = prepare_image(model, image)
    embedding, intermediate = encode_image(model, tensor, family)
    height, width = mask.shape
    box = box_prompt(mask, prompt_source)
    box *= np.array([input_hw[1] / width, input_hw[0] / height] * 2, dtype=np.float32)
    boxes = torch.as_tensor(box, device=model.device).reshape(1, 4)
    logits, _ = decode(model, embedding, intermediate, family, boxes=boxes)
    # This official helper uses differentiable interpolation/cropping.
    return model.postprocess_masks(logits, input_hw, (height, width))


@torch.no_grad()
def automatic_proposals(model, image, family, settings):
    """Original SAM AMG or the notebook's HQ point-grid plus mask-IoU NMS."""
    if family == "medsam":
        raise ValueError("MedSAM automatic/tiled inference was not implemented in the supplied source")
    if family == "sam":
        from segment_anything import SamAutomaticMaskGenerator
        keys = ("points_per_side", "pred_iou_thresh", "stability_score_thresh",
                "min_mask_region_area", "crop_n_layers", "points_per_batch")
        generator = SamAutomaticMaskGenerator(model, **{k: settings[k] for k in keys if k in settings})
        return generator.generate(image)
    tensor, input_hw = prepare_image(model, image)
    embedding, intermediate = encode_image(model, tensor, family)
    side = settings.get("points_per_side", 32)
    xs = torch.linspace(0, max(1, input_hw[1]-1), side, device=model.device)
    ys = torch.linspace(0, max(1, input_hw[0]-1), side, device=model.device)
    candidates = []
    for y in ys:
        for x in xs:
            point = torch.stack([x, y]).reshape(1, 1, 2)
            labels = torch.ones((1, 1), device=model.device, dtype=torch.int64)
            logits, score = decode(model, embedding, intermediate, family, points=(point, labels))
            probability = model.postprocess_masks(logits, input_hw, image.shape[:2]).sigmoid()[0, 0].cpu().numpy()
            binary = probability > settings.get("threshold", 0.5)
            quality = float(score[0, 0])
            if binary.sum() >= settings.get("min_mask_region_area", 60) and quality >= settings.get("pred_iou_thresh", 0.8):
                candidates.append({"segmentation": binary, "area": int(binary.sum()), "score": quality})
    kept = []
    for candidate in sorted(candidates, key=lambda row: row["score"], reverse=True):
        binary = candidate["segmentation"]
        overlaps = [(binary & item["segmentation"]).sum() / max(1, (binary | item["segmentation"]).sum())
                    for item in kept]
        if not any(iou >= settings.get("nms_iou_threshold", 0.4) for iou in overlaps):
            kept.append(candidate)
    return kept
