"""Save numeric masks, score arrays and provenance for evaluation and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from froth.config import SAM_MODELS
from froth.data.dataset import image_tensor
from froth.data.labelme import load_sample, prediction_name, read_samples, transform_pair
from froth.io import provenance, sha256, write_csv, write_json
from froth.models.factory import build_model, foreground_probability, load_checkpoint, load_model_weights, logits_from_output
from froth.models.sam_family import automatic_proposals, prompted_logits
from .training import select_device


def require_compatible_checkpoint(config, checkpoint):
    saved = load_checkpoint(checkpoint)
    if isinstance(saved, dict) and "config" in saved:
        old = saved["config"]
        for key in ("name", "variant", "encoder", "output_stride", "output_channels"):
            if old["model"].get(key) != config["model"].get(key):
                raise ValueError(f"Checkpoint/config model mismatch at {key}")
        if old["data"] != config["data"] or old["training"]["loss"] != config["training"]["loss"]:
            raise ValueError("Checkpoint/config preprocessing or output-loss mismatch")
    return saved.get("training_run_sha256") if isinstance(saved, dict) else None


@torch.no_grad()
def predict(config, dataset_root, manifest, output, checkpoint=None, *, split="test", device="auto",
            limit=None, mode="semantic", prompt_source=None, use_base_weights=False):
    family = config["model"]["name"]
    sam = family in SAM_MODELS
    if mode == "amg" and family not in {"sam", "hq_sam"}:
        raise ValueError("AMG mode is supported by the supplied SAM and HQ-SAM source only")
    if sam and mode == "semantic" and prompt_source not in {"full_image", "ground_truth"}:
        raise ValueError("Set --prompt-source full_image or ground_truth explicitly for prompted SAM evaluation")
    if not checkpoint and not (sam and use_base_weights):
        raise ValueError("Provide --checkpoint, or explicitly --use-base-weights for a SAM-family base model")
    training_run_hash = require_compatible_checkpoint(config, checkpoint) if checkpoint else None
    output = Path(output)
    if (output / "run.json").exists():
        raise FileExistsError(f"Prediction output already contains a run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    (output / "scores").mkdir(exist_ok=True)
    samples = read_samples(dataset_root, manifest, split)[:limit]
    device = select_device(device)
    model = build_model(config, pretrained=False).to(device).eval()
    if checkpoint:
        load_model_weights(model, checkpoint)
    metadata = provenance(config, manifest, device=str(device), inference_mode=mode,
                          prompt_source=prompt_source if mode == "semantic" and sam else "none",
                          ground_truth_assisted=bool(sam and mode == "semantic" and prompt_source == "ground_truth"),
                          weights_source="fine_tuned_checkpoint" if checkpoint else "base_checkpoint",
                          score_kind="binary_union_of_proposals" if mode == "amg" else "foreground_probability")
    if checkpoint:
        metadata["checkpoint_sha256"] = sha256(checkpoint)
        metadata["training_run_sha256"] = training_run_hash
    if config["model"].get("base_checkpoint"):
        metadata["base_checkpoint_sha256"] = sha256(config["model"]["base_checkpoint"])
    records, geometries = [], {}
    for index, sample in enumerate(samples, 1):
        image, mask = load_sample(sample)
        image, mask, geometry = transform_pair(image, mask, config["data"])
        instances = np.zeros(mask.shape, dtype=np.int32)
        proposal_count = 0
        if sam and mode == "amg":
            proposals = automatic_proposals(model, image, family, config["inference"])
            # Overlaps use first/highest-score ownership; union metrics do not
            # depend on this display-only instance assignment.
            proposals.sort(key=lambda row: row.get("predicted_iou", row.get("score", 0)), reverse=True)
            for label, proposal in enumerate(proposals, 1):
                instances[(instances == 0) & proposal["segmentation"]] = label
            proposal_count = len(proposals)
            probability = (instances > 0).astype(np.float32)
        elif sam:
            logits = prompted_logits(model, image, mask, family, prompt_source)
            probability = logits.sigmoid()[0, 0].cpu().numpy()
        else:
            tensor = image_tensor(image, config["data"].get("normalize", False))[None].to(device)
            logits = logits_from_output(model(tensor))
            if logits.shape[-2:] != mask.shape:
                logits = torch.nn.functional.interpolate(logits, mask.shape, mode="bilinear", align_corners=False)
            probability = foreground_probability(logits, config["training"]["loss"])[0].cpu().numpy()
        binary = probability > config["inference"]["threshold"]
        filename = prediction_name(sample.image_name)
        Image.fromarray(binary.astype(np.uint8) * 255).save(output / "masks" / filename)
        np.savez_compressed(output / "scores" / (sample.image_name + ".npz"),
                            probability=probability.astype(np.float32), instances=instances)
        geometries[sample.image_name] = geometry
        records.append({"image_name": sample.image_name, "source_split": sample.split,
                        "height": mask.shape[0], "width": mask.shape[1], "proposal_count": proposal_count,
                        "image_sha256": sha256(sample.image_path), "annotation_sha256": sha256(sample.annotation_path),
                        "mask_file": f"masks/{filename}", "score_file": f"scores/{sample.image_name}.npz"})
        print(f"Predicted {index}/{len(samples)}: {sample.image_name}", flush=True)
    metadata["images"] = len(records)
    metadata["geometry"] = geometries
    write_json(output / "run.json", metadata)
    write_csv(output / "predictions.csv", records)
    return output
