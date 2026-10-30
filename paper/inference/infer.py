"""GPU inference only, with the stored checkpoints and the public zero-shot weights; no training.

Semantic models write a foreground probability map (uint8 PNG, 0-255) per image.
Automatic (AMG-style) models write an instance label map (uint16 PNG).
Every model reproduces the preprocessing of the notebook that trained it,
including the quirks (unnormalised SAM input, anisotropic box scaling, MedSAM pad-then-resize).

usage: python -m paper.inference.infer --split test1 [--models dlv3,dlv3p,...] [--list names.txt] [--amg]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide

from paper.paths import CKPT as CK
from paper.paths import DATA, ROBUSTSAM
from paper.paths import RUN as OUT

DEV = torch.device("cuda")
MEAN_IMNET, STD_IMNET = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
SAM_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
SAM_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


def load_rgb(path):
    """HWC uint8 RGB, as both notebook loaders produce for these 8-bit TIFFs (PNG copies are pixel-identical)."""
    if str(path).lower().endswith(".png"):
        return np.ascontiguousarray(
            cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        )
    a = np.asarray(tifffile.imread(str(path)))
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[0] != a.shape[-1]:
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[2] == 1:
        a = np.repeat(a, 3, 2)
    a = a[:, :, :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        a = (255 * (a - a.min()) / max(1e-6, a.max() - a.min())).astype(np.uint8)
    return np.ascontiguousarray(a)


def images(split, names=None):
    d = DATA / split
    files = sorted(
        p
        for p in d.iterdir()
        if p.suffix.lower() in (".tif", ".tiff", ".png") and not p.name.startswith(".")
    )
    if names:
        files = [
            p for p in files if p.name in names or p.stem in {n.rsplit(".", 1)[0] for n in names}
        ]
    return files


# ---------------------------------------------------------------- CNNs
def build_dlv3():
    from torchvision.models.segmentation import deeplabv3_resnet101
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead

    m = deeplabv3_resnet101(weights=None, weights_backbone=None, num_classes=21, aux_loss=True)
    m.classifier = DeepLabHead(2048, 2)
    m.load_state_dict(
        torch.load(CK / "dlv3_r101_ep9.pth", map_location="cpu")["model_state"], strict=True
    )
    return m


def build_dlv3p():
    import segmentation_models_pytorch as smp

    m = smp.DeepLabV3Plus(
        encoder_name="resnet152",
        encoder_weights=None,
        encoder_output_stride=8,
        in_channels=3,
        classes=2,
    )
    sd = torch.load(CK / "dlv3p_r152_ep48.pth", map_location="cpu")["model_state"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"DLV3+ key mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    return m


@torch.no_grad()
def cnn_prob(model, img):
    H, W = img.shape[:2]
    x = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - MEAN_IMNET) / STD_IMNET
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(DEV)
    out = model(x)
    out = out["out"] if isinstance(out, dict) else out
    p = out.softmax(1)[:, 1:2]
    return (
        F.interpolate(p, (H, W), mode="bilinear", align_corners=False)[0, 0].float().cpu().numpy()
    )


# ---------------------------------------------------------------- SAM family
RLS = ResizeLongestSide(1024)


def sam_input(img, normalize):
    r = RLS.apply_image(img)
    h, w = r.shape[:2]
    t = torch.as_tensor(r).permute(2, 0, 1).float()[None]
    if normalize:  # MedSAM notebook: pad with zeros first, then normalise the whole canvas
        t = F.pad(t, (0, 1024 - w, 0, 1024 - h))
        t = (t - SAM_MEAN) / SAM_STD
    else:  # SAM/HQ notebooks: /255, no mean/std, zero pad
        t = F.pad(t / 255.0, (0, 1024 - w, 0, 1024 - h))
    return t.to(DEV), (h, w)


def box_aniso(H, W):  # SAM/HQ notebooks: x*1024/W, y*1024/H (covers the padding too)
    return torch.tensor([[[0.0, 0.0, (W - 1) * 1024.0 / W, (H - 1) * 1024.0 / H]]], device=DEV)


def box_iso(H, W):  # MedSAM notebook: aspect-preserving scale of the box
    s = min(1024.0 / H, 1024.0 / W)
    return torch.tensor([[[0.0, 0.0, min(1024.0, W * s), min(1024.0, H * s)]]], device=DEV)


def up_crop(low, hw, HW):  # SAM/HQ notebooks: 256 -> 1024, crop valid window, -> original
    u = F.interpolate(low, (1024, 1024), mode="bilinear", align_corners=False)[
        ..., : hw[0], : hw[1]
    ]
    return F.interpolate(u, HW, mode="bilinear", align_corners=False)


def up_padstretch(
    low, hw, HW
):  # MedSAM notebook: 256 -> (h,w), zero-pad to 1024, stretch canvas -> original
    u = F.interpolate(low, hw, mode="bilinear", align_corners=False)
    u = F.pad(u, (0, 1024 - hw[1], 0, 1024 - hw[0]))
    return F.interpolate(u, HW, mode="bilinear", align_corners=False)


def build_sam(decoder=None):
    from segment_anything import sam_model_registry

    m = sam_model_registry["vit_b"](checkpoint=str(CK / "sam_vit_b_01ec64.pth"))
    if decoder:
        m.mask_decoder.load_state_dict(
            torch.load(CK / decoder, map_location="cpu")["mask_decoder"], strict=True
        )
    return m


def build_hq(decoder=None):
    from segment_anything_hq import sam_model_registry

    m = sam_model_registry["vit_b"](checkpoint=str(CK / "sam_hq_vit_b.pth"))
    if decoder:
        m.mask_decoder.load_state_dict(
            torch.load(CK / decoder, map_location="cpu")["mask_decoder"], strict=True
        )
    return m


def build_medsam():
    from segment_anything import sam_model_registry

    m = sam_model_registry["vit_b"](checkpoint=None)
    m.load_state_dict(
        torch.load(CK / "medsam_vit_b_decoder_only_full_best.pth", map_location="cpu"), strict=True
    )
    return m


def build_robust():
    sys.path.insert(0, str(ROBUSTSAM))
    from robust_segment_anything import sam_model_registry

    return sam_model_registry["vit_b"](
        opt=None, checkpoint=str(CK / "model_checkpoint" / "robustsam_checkpoint_b.pth")
    )


@torch.no_grad()
def sam_box_prob(m, img, kind):
    H, W = img.shape[:2]
    if kind == "robust":
        t = torch.as_tensor(img).permute(2, 0, 1)[None].to(DEV).float()
        t = RLS.apply_image_torch(t)
        box = RLS.apply_boxes_torch(
            torch.tensor([[0, 0, W - 1, H - 1]], device=DEV, dtype=torch.float), (H, W)
        )
        o = m.predict(
            None,
            [{"image": t, "boxes": box[None], "original_size": (H, W)}],
            multimask_output=False,
            return_logits=True,
        )
        return torch.sigmoid(o[0]["masks"][0, 0].float()).cpu().numpy()
    x, hw = sam_input(img, normalize=(kind == "medsam"))
    if kind == "hq":
        emb, interm = m.image_encoder(x)
    else:
        emb = m.image_encoder(x)
    box = box_iso(H, W) if kind == "medsam" else box_aniso(H, W)
    sp, de = m.prompt_encoder(points=None, boxes=box, masks=None)
    kw = dict(
        image_embeddings=emb,
        image_pe=m.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sp,
        dense_prompt_embeddings=de,
        multimask_output=False,
    )
    if kind == "hq":
        kw.update(hq_token_only=True, interm_embeddings=interm)
    low = m.mask_decoder(**kw)[0]
    up = up_padstretch(low, hw, (H, W)) if kind == "medsam" else up_crop(low, hw, (H, W))
    return torch.sigmoid(up)[0, 0].float().cpu().numpy()


# ---------------------------------------------------------------- automatic (instance) modes
def to_labels(masks, scores, HW):
    lab = np.zeros(HW, np.uint16)
    for k, i in enumerate(np.argsort(-np.asarray(scores)), 1):
        lab[(lab == 0) & masks[i]] = k
    return lab


def amg_official(m, img, robust=False, **kw):
    if robust:
        from robust_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
    else:
        from segment_anything import SamAutomaticMaskGenerator
    g = SamAutomaticMaskGenerator(m, **kw)
    res = g.generate(img)
    return to_labels(
        [r["segmentation"] for r in res], [r["predicted_iou"] for r in res], img.shape[:2]
    ), len(res)


@torch.no_grad()
def amg_hq_grid(m, img, pps=32, prob_thr=0.5, iou_thr=0.80, min_area=60, nms=0.32, batch=64):
    """Batched, otherwise identical re-implementation of the 2HQSAM notebook's hq_amg_generate()."""
    H, W = img.shape[:2]
    x, (h, w) = sam_input(img, normalize=False)
    emb, interm = m.image_encoder(x)
    xs = torch.linspace(0, max(1, w - 1), pps, device=DEV)
    ys = torch.linspace(0, max(1, h - 1), pps, device=DEV)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    pe = m.prompt_encoder.get_dense_pe()
    keep_m, keep_s = [], []
    for b in range(0, len(pts), batch):
        p = pts[b : b + batch]
        n = len(p)
        sp, de = m.prompt_encoder(
            points=(p[:, None, :], torch.ones((n, 1), device=DEV, dtype=torch.int)),
            boxes=None,
            masks=None,
        )
        low, iou = m.mask_decoder(
            image_embeddings=emb,
            image_pe=pe,
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=de,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=interm,
        )
        prob = torch.sigmoid(up_crop(low, (h, w), (H, W)))[:, 0] > prob_thr
        area = prob.flatten(1).sum(1)
        sc = iou[:, 0]
        ok = (area >= min_area) & (sc >= iou_thr)
        keep_m.append(prob[ok])
        keep_s.append(sc[ok])
    M = torch.cat(keep_m)
    S = torch.cat(keep_s)
    order = torch.argsort(S, descending=True)
    M = M[order]
    S = S[order]
    if len(M) == 0:
        return np.zeros((H, W), np.uint16), 0
    Mf = M.flatten(1).half()
    inter = Mf @ Mf.T
    a = Mf.sum(1)
    iou_m = (inter / (a[:, None] + a[None, :] - inter + 1e-6)).float().cpu().numpy()
    kept = []
    for i in range(len(M)):
        if all(iou_m[i, j] < nms for j in kept):
            kept.append(i)
    Mk = M[kept].cpu().numpy()
    Sk = S[kept].cpu().numpy()
    return to_labels(list(Mk), Sk, (H, W)), len(kept)


@torch.no_grad()
def amg_robust_grid(m, img, pps=32, iou_thr=0.0, min_area=100, nms=0.32, batch=1):
    """RobustSAM point grid through its official single-prompt decoder path (clear=False, as in RobustSAM.predict()).
    The repository's SamAutomaticMaskGenerator calls the decoder with clear=True, which returns empty masks."""
    H, W = img.shape[:2]
    t = RLS.apply_image_torch(torch.as_tensor(img).permute(2, 0, 1)[None].to(DEV).float())
    x = m.preprocess(t)
    emb, feats = m.image_encoder(x)
    enc = feats[0][0].unsqueeze(0).unsqueeze(0)
    h, w = t.shape[-2:]
    xs = torch.linspace(0, W - 1, pps, device=DEV)
    ys = torch.linspace(0, H - 1, pps, device=DEV)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    pts = RLS.apply_coords_torch(pts, (H, W))
    pe = m.prompt_encoder.get_dense_pe()
    keep_m, keep_s = [], []
    for k in range(len(pts)):
        sp, de = m.prompt_encoder(
            points=(pts[k].view(1, 1, 2), torch.ones((1, 1), device=DEV, dtype=torch.int)),
            boxes=None,
            masks=None,
        )
        low, iou, _, _ = m.mask_decoder(
            image_embeddings=emb,
            image_pe=pe,
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=de,
            multimask_output=False,
            encoder_features=enc,
            clear=False,
        )
        mk = m.postprocess_masks(low[:, :1], (h, w), (H, W))[0, 0] > m.mask_threshold
        if mk.sum() >= min_area and float(iou[0, 0]) >= iou_thr:
            keep_m.append(mk)
            keep_s.append(float(iou[0, 0]))
    if not keep_m:
        return np.zeros((H, W), np.uint16), 0
    order = np.argsort(keep_s)[::-1]
    M = torch.stack([keep_m[i] for i in order])
    S = [keep_s[i] for i in order]
    Mf = M.flatten(1).half()
    inter = Mf @ Mf.T
    a = Mf.sum(1)
    iou_m = (inter / (a[:, None] + a[None, :] - inter + 1e-6)).float().cpu().numpy()
    kept = []
    for i in range(len(M)):
        if all(iou_m[i, j] < nms for j in kept):
            kept.append(i)
    return to_labels(list(M[kept].cpu().numpy()), [S[i] for i in kept], (H, W)), len(kept)


# ---------------------------------------------------------------- zero-shot instance baselines and corrected variants
AMG_KW = dict(
    points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92, min_mask_region_area=80
)


def build_sam_h():
    from segment_anything import sam_model_registry

    return sam_model_registry["vit_h"](checkpoint=str(CK / "sam_vit_h_4b8939.pth"))


def build_hq_h():
    from segment_anything_hq import sam_model_registry

    return sam_model_registry["vit_h"](checkpoint=str(CK / "sam_hq_vit_h.pth"))


def amg_hq_official(m, img):
    from segment_anything_hq import SamAutomaticMaskGenerator

    res = SamAutomaticMaskGenerator(m, **AMG_KW).generate(img)
    return to_labels(
        [r["segmentation"] for r in res], [r["predicted_iou"] for r in res], img.shape[:2]
    ), len(res)


def build_sam2():
    from sam2.build_sam import build_sam2 as b

    return b("configs/sam2.1/sam2.1_hiera_l.yaml", str(CK / "sam2.1_hiera_large.pt"), device="cuda")


def amg_sam2(m, img):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    res = SAM2AutomaticMaskGenerator(m, **AMG_KW).generate(img)
    return to_labels(
        [r["segmentation"] for r in res], [r["predicted_iou"] for r in res], img.shape[:2]
    ), len(res)


class CellposeWrap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from cellpose import models

        self.cp = models.CellposeModel(gpu=True)  # default generalist model (Cellpose-SAM)

    def forward(self, *a):
        raise NotImplementedError


def run_cellpose(w, img):
    masks = w.cp.eval(img, batch_size=8)[0]
    return masks.astype(np.uint16), int(masks.max())


def build_sam_ft_rawinput():
    """Fine-tuned SAM decoder with the automatic generator fed the SAME input scaling as during fine-tuning
    (values /255, zero padding, no SAM mean/std) instead of the generator's internal normalisation."""
    m = build_sam("sam_vit_b_decoder_only_decoder_best.pth")

    def pre(x):
        h, w = x.shape[-2:]
        x = x / 255.0
        return F.pad(x, (0, m.image_encoder.img_size - w, 0, m.image_encoder.img_size - h))

    m.preprocess = pre
    return m


@torch.no_grad()
def medsam_fixed_prob(m, img):
    """MedSAM with the standard up-sampling (crop the valid window) instead of the training-time pad-and-stretch."""
    H, W = img.shape[:2]
    x, hw = sam_input(img, normalize=True)
    emb = m.image_encoder(x)
    sp, de = m.prompt_encoder(points=None, boxes=box_iso(H, W), masks=None)
    low = m.mask_decoder(
        image_embeddings=emb,
        image_pe=m.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sp,
        dense_prompt_embeddings=de,
        multimask_output=False,
    )[0]
    return torch.sigmoid(up_crop(low, hw, (H, W)))[0, 0].float().cpu().numpy()


# ---------------------------------------------------------------- registry
SEMANTIC = {
    "dlv3": (build_dlv3, "cnn"),
    "dlv3p": (build_dlv3p, "cnn"),
    "sam_ft": (lambda: build_sam("sam_vit_b_decoder_only_decoder_best.pth"), "sam"),
    "hq_ft": (lambda: build_hq("sam_hq_vit_b_decoder_best.pth"), "hq"),
    "medsam_ft": (build_medsam, "medsam"),
    "sam_zs": (lambda: build_sam(None), "sam"),
    "robust_zs": (build_robust, "robust"),
    "medsam_ft_fix": (build_medsam, "medsam_fix"),
}
AMG = {
    "amg_sam_zs": (
        lambda: build_sam(None),
        lambda m, im: amg_official(
            m,
            im,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=80,
        ),
    ),
    "amg_sam_ft": (
        lambda: build_sam("sam_vit_b_decoder_only_decoder_best.pth"),
        lambda m, im: amg_official(
            m,
            im,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=80,
        ),
    ),
    "amg_hq_ft": (lambda: build_hq("sam_hq_vit_b_decoder_best.pth"), amg_hq_grid),
    "amg_robust_zs": (build_robust, amg_robust_grid),
    "amg_sam_h_zs": (build_sam_h, lambda m, im: amg_official(m, im, **AMG_KW)),
    "amg_hq_h_zs": (build_hq_h, amg_hq_official),
    "amg_sam2_zs": (build_sam2, amg_sam2),
    "cellpose_zs": (CellposeWrap, run_cellpose),
    "amg_sam_ft_raw": (build_sam_ft_rawinput, lambda m, im: amg_official(m, im, **AMG_KW)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--models", default="")
    ap.add_argument("--list", default=None)
    ap.add_argument("--amg", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    names = set(Path(a.list).read_text().split("\n")) - {""} if a.list else None
    files = images(a.split, names)
    reg = AMG if a.amg else SEMANTIC
    todo = [k for k in (a.models.split(",") if a.models else reg)]
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "infer_log.jsonl", "a")
    for key in todo:
        build, kind = reg[key]
        t0 = time.time()
        m = build()
        m = m.to(DEV).eval() if hasattr(m, "to") and not isinstance(m, CellposeWrap) else m
        print(f"[{key}] built in {time.time() - t0:.1f}s", flush=True)
        d = OUT / key / a.split
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            dst = d / (f.stem + ".png")
            if dst.exists() and not a.overwrite:
                continue
            img = load_rgb(f)
            t1 = time.time()
            if a.amg:
                lab, n = kind(m, img)
                cv2.imwrite(str(dst), lab)
                rec = dict(
                    model=key,
                    split=a.split,
                    image=f.name,
                    n_proposals=int(n),
                    sec=round(time.time() - t1, 3),
                )
            elif kind == "cnn":
                p = cnn_prob(m, img)
                cv2.imwrite(str(dst), np.rint(p * 255).astype(np.uint8))
                rec = dict(model=key, split=a.split, image=f.name, sec=round(time.time() - t1, 3))
            elif kind == "medsam_fix":
                p = medsam_fixed_prob(m, img)
                cv2.imwrite(str(dst), np.rint(p * 255).astype(np.uint8))
                rec = dict(model=key, split=a.split, image=f.name, sec=round(time.time() - t1, 3))
            else:
                p = sam_box_prob(m, img, kind)
                cv2.imwrite(str(dst), np.rint(p * 255).astype(np.uint8))
                rec = dict(model=key, split=a.split, image=f.name, sec=round(time.time() - t1, 3))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if i % 20 == 0:
                print(f"[{key}] {i + 1}/{len(files)} {f.name} {rec['sec']}s", flush=True)
        del m
        torch.cuda.empty_cache()
    print("STAGE_A_DONE", flush=True)


if __name__ == "__main__":
    main()
