"""Historical source export; maintained entry points live in scripts/.

Original: codes/MEDSAM/3MEDSAM.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    import os
    os.getcwd()

    # %% Original cell 1 (code)
    # ========= MedSAM from scratch — Cell 1: paths + dataset (LabelMe) =========
    from pathlib import Path
    import json
    import numpy as np
    import cv2
    import tifffile as tiff

    # ----- EDIT THESE PATHS IF NEEDED -----
    ROOT     = Path("F:/dataset")
    TRAINDIR = ROOT / "train1"
    TESTDIR  = ROOT / "test1"
    WEIGHTS  = ROOT / "froth_finetune" / "weights" / "medsam_vit_b.pth"   # <-- put your MedSAM checkpoint here
    OUTDIR   = ROOT / "frtoh_finetune" / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    assert TRAINDIR.exists(), f"Missing folder: {TRAINDIR}"
    assert TESTDIR.exists(),  f"Missing folder: {TESTDIR}"

    # ----------------- helpers -----------------
    def load_uint8_rgb(path: Path) -> np.ndarray:
        """Read TIFF, return HWC uint8 RGB (drops alpha, scales >8-bit to 0..255)."""
        im = tiff.imread(str(path))
        im = np.array(im)
        # CHW -> HWC if needed
        if im.ndim == 3 and im.shape[0] in (1,3,4) and im.shape[0] != im.shape[2]:
            im = np.transpose(im, (1,2,0))
        # Gray -> 3-ch
        if im.ndim == 2:
            im = np.stack([im]*3, axis=-1)
        # Clip/scale to 8-bit
        if im.dtype != np.uint8:
            mx = float(im.max()) if im.max() != 0 else 1.0
            im = np.clip((im.astype(np.float32) / mx) * 255.0, 0, 255).astype(np.uint8)
        # Drop alpha if present
        if im.shape[2] >= 4:
            im = im[:, :, :3]
        return im

    def labelme_to_mask(json_path: Path, hw: tuple[int,int], label_name: str = "froth") -> np.ndarray:
        """Make binary mask (H,W) uint8 (0/255) from LabelMe JSON for a given label."""
        H, W = hw
        mask = np.zeros((H, W), dtype=np.uint8)
        with open(json_path, "r") as f:
            data = json.load(f)
        for shp in data.get("shapes", []):
            if shp.get("label", "") != label_name:
                continue
            pts = np.array(shp.get("points", []), dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 3:
                cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        return mask

    def mask_to_xyxy(mask: np.ndarray) -> list[int] | None:
        """Tight bbox around positive mask, or None if empty."""
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        return [x0, y0, x1, y1]

    # ----------------- dataset -----------------
    class FrothSegmentationDataset:
        """
        Each item:
          - image_np: HWC uint8 RGB
          - mask_np:  HxW uint8 (0/255)
          - orig_hw:  (H, W) ints
          - bbox_xyxy: [x0,y0,x1,y1] or None
          - img_path/json_path: str (for debugging)
        """
        def __init__(self, folder: Path, label_name: str = "froth"):
            self.folder = Path(folder)
            self.label_name = label_name
            self.items = []
            imgs = list(self.folder.glob("*.tif")) + list(self.folder.glob("*.tiff"))
            for img in sorted(imgs):
                js = img.with_suffix(".json")
                if js.exists():
                    self.items.append((img, js))
            if not self.items:
                raise RuntimeError(f"No paired .tif/.tiff + .json in {self.folder}")

        def __len__(self): return len(self.items)

        def __getitem__(self, idx: int) -> dict:
            img_path, json_path = self.items[idx]
            image = load_uint8_rgb(img_path)
            H, W = image.shape[:2]
            mask = labelme_to_mask(json_path, (H, W), self.label_name)
            bbox = mask_to_xyxy(mask)
            return {
                "image_np": image,
                "mask_np":  mask,
                "orig_hw":  (H, W),
                "bbox_xyxy": bbox,
                "img_path": str(img_path),
                "json_path":str(json_path),
            }

    # Quick sanity print
    train_ds = FrothSegmentationDataset(TRAINDIR, label_name="froth")
    test_ds  = FrothSegmentationDataset(TESTDIR,  label_name="froth")
    print(f"train: {len(train_ds)} | test: {len(test_ds)}")

    s = train_ds[0]
    print("sample:", s["image_np"].shape, s["image_np"].dtype, "|", s["mask_np"].shape, s["mask_np"].dtype, "| bbox:", s["bbox_xyxy"])

    # %% Original cell 2 (code)
    # ========= MedSAM from scratch — Cell 2: load model + decoder-only setup =========
    import torch
    from torch import nn
    from segment_anything import sam_model_registry

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_TYPE = "vit_b"  # MedSAM is a ViT-B variant

    def load_medsam_vit_b(weights_path: Path, device: torch.device):
        """
        Robust loader:
        - Try registry with checkpoint directly.
        - Fallback: instantiate then load a state_dict from various common keys.
        Returns model and (missing, unexpected) keys from load (for inspection).
        """
        # 1) try the "native" way
        try:
            model = sam_model_registry[MODEL_TYPE](checkpoint=str(weights_path))
            model.to(device)
            return model, (), ()
        except Exception as e_native:
            # 2) fallback to manual state_dict load
            model = sam_model_registry[MODEL_TYPE]()  # init w/o weights
            sd = torch.load(str(weights_path), map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                state = sd["state_dict"]
            elif isinstance(sd, dict) and "model" in sd:
                state = sd["model"]
            else:
                state = sd  # assume raw state_dict

            # Some checkpoints store keys with module prefixes; load non-strict to be safe
            missing, unexpected = model.load_state_dict(state, strict=False)
            model.to(device)
            print("[load_medsam_vit_b] Fallback loader used.")
            if missing:
                print("  Missing keys:", len(missing))
            if unexpected:
                print("  Unexpected keys:", len(unexpected))
            return model, missing, unexpected

    sam, missing_keys, unexpected_keys = load_medsam_vit_b(WEIGHTS, DEVICE)

    # ---- decoder-only finetune ----
    def set_decoder_only(model: nn.Module):
        # freeze everything
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze decoder (and IoU head inside it)
        for n, p in model.named_parameters():
            if n.startswith("mask_decoder."):
                p.requires_grad = True
        if hasattr(model, "mask_decoder") and hasattr(model.mask_decoder, "iou_prediction_head"):
            for p in model.mask_decoder.iou_prediction_head.parameters():
                p.requires_grad = True
        model.train()
        return model

    sam = set_decoder_only(sam)

    # ---- summary ----
    total_params     = sum(p.numel() for p in sam.parameters())
    trainable_params = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    print(f"Device: {DEVICE}")
    print(f"Loaded MedSAM weights: {WEIGHTS.name}")
    print(f"Total params: {total_params/1e6:.2f}M | Trainable (decoder-only): {trainable_params/1e6:.2f}M")

    print("First trainable parameters (should all be under 'mask_decoder.*'):")
    shown = 0
    for n, p in sam.named_parameters():
        if p.requires_grad:
            print(f"  {n}  {tuple(p.shape)}")
            shown += 1
            if shown >= 12:
                break

    # %% Original cell 3 (code)
    # ========= MedSAM from scratch — Cell 3: per-sample decoder-only training =========
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from tqdm.auto import tqdm
    from segment_anything.utils.transforms import ResizeLongestSide

    # ----- training config -----
    IMAGE_LONGEST = 1024
    EPOCHS        = 1          # start small, you can increase later
    LR            = 1e-4
    WEIGHT_DECAY  = 0.01
    AMP           = torch.cuda.is_available()  # autocast only if CUDA is present

    # ----- transforms & constants -----
    resize_longest = ResizeLongestSide(IMAGE_LONGEST)
    sam_pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=DEVICE).view(1,3,1,1)
    sam_pixel_std  = torch.tensor([58.395, 57.12, 57.375], device=DEVICE).view(1,3,1,1)

    bce = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    # ----- helpers -----
    def prepare_image_for_sam(image_np: np.ndarray):
        """
        HWC uint8 -> (1,3,1024,1024) float on DEVICE, plus sizes:
          - input_hw = (Hin, Win) after ResizeLongestSide
          - orig_hw  = (H0, W0)
        """
        H0, W0 = image_np.shape[:2]
        img_resized = resize_longest.apply_image(image_np)       # (Hin, Win, 3) uint8
        Hin, Win = img_resized.shape[:2]
        img_3chw = torch.as_tensor(img_resized).permute(2,0,1).float()  # (3,Hin,Win)
        pad_h = IMAGE_LONGEST - Hin
        pad_w = IMAGE_LONGEST - Win
        img_3chw = F.pad(img_3chw, (0, pad_w, 0, pad_h))        # (3,1024,1024)
        img_3chw = img_3chw.unsqueeze(0).to(DEVICE)             # (1,3,1024,1024)
        return img_3chw, (Hin, Win), (H0, W0)

    def resize_box_to_1024(box_xyxy, orig_hw):
        """Map bbox from original (H0,W0) to 1024 canvas coords (top-left padded)."""
        if box_xyxy is None:
            return [0.0, 0.0, float(IMAGE_LONGEST), float(IMAGE_LONGEST)]
        x0,y0,x1,y1 = [float(v) for v in box_xyxy]
        H0,W0 = orig_hw
        scale = min(IMAGE_LONGEST / H0, IMAGE_LONGEST / W0)
        x0r = max(0.0, min(IMAGE_LONGEST-1.0, x0 * scale))
        y0r = max(0.0, min(IMAGE_LONGEST-1.0, y0 * scale))
        x1r = max(0.0, min(float(IMAGE_LONGEST), x1 * scale))
        y1r = max(0.0, min(float(IMAGE_LONGEST), y1 * scale))
        return [x0r, y0r, x1r, y1r]

    def postprocess_masks_to_original(lowres_pred, input_hw, orig_hw):
        """
        lowres_pred: (1,1,256,256) logits
        input_hw:    (Hin, Win) ints
        orig_hw:     (H0, W0) ints
        returns:     (1,1,H0,W0) logits
        """
        Hin, Win = int(input_hw[0]), int(input_hw[1])
        H0,  W0  = int(orig_hw[0]),  int(orig_hw[1])
        up = F.interpolate(lowres_pred, size=(Hin, Win), mode="bilinear", align_corners=False)
        pad_h = IMAGE_LONGEST - Hin
        pad_w = IMAGE_LONGEST - Win
        canvas = F.pad(up, (0, pad_w, 0, pad_h))                # (1,1,1024,1024)
        out = F.interpolate(canvas, size=(H0, W0), mode="bilinear", align_corners=False)
        return out

    def iou_from_logits(logits, target_bin, thresh=0.5, eps=1e-6):
        """logits (1,1,H,W), target_bin (1,1,H,W) in {0,1}."""
        pred = (torch.sigmoid(logits) > thresh).float()
        inter = (pred * target_bin).sum()
        union = pred.sum() + target_bin.sum() - inter
        return float(((inter + eps) / (union + eps)).item())

    # ----- optimizer on decoder-only params -----
    optimizer = torch.optim.AdamW([p for p in sam.parameters() if p.requires_grad],
                                  lr=LR, weight_decay=WEIGHT_DECAY)

    dec_best_path  = OUTDIR / "medsam_vit_b_decoder_only_decoder_best.pth"
    full_best_path = OUTDIR / "medsam_vit_b_decoder_only_full_best.pth"
    best_mIoU = -1.0

    print("Starting training (decoder-only), per-sample without DataLoader...")
    for epoch in range(1, EPOCHS+1):
        sam.train()  # only decoder has requires_grad=True
        running_loss = 0.0

        for idx in tqdm(range(len(train_ds)), desc=f"[Epoch {epoch}/{EPOCHS}] train"):
            sample = train_ds[idx]
            img_np  = sample["image_np"]
            msk_np  = sample["mask_np"]
            orig_hw = sample["orig_hw"]
            box_xyxy= resize_box_to_1024(sample["bbox_xyxy"], orig_hw)

            # 1) prepare tensors
            img_in, input_hw, orig_hw = prepare_image_for_sam(img_np)
            img_in = (img_in - sam_pixel_mean) / sam_pixel_std        # normalize (1,3,1024,1024)
            target = torch.from_numpy((msk_np > 127).astype(np.float32))[None,None,...].to(DEVICE)  # (1,1,H,W)

            # 2) frozen encoders
            with torch.no_grad():
                img_embed = sam.image_encoder(img_in)                 # (1,256,64,64)
                dense_pe  = sam.prompt_encoder.get_dense_pe().to(DEVICE)  # (1,256,64,64)
                boxes_1024= torch.tensor(box_xyxy, device=DEVICE, dtype=torch.float32).view(1,1,4)
                sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=boxes_1024, masks=None)

            # 3) trainable decoder
            with torch.cuda.amp.autocast(enabled=AMP):
                lowres_logits, _ = sam.mask_decoder(
                    image_embeddings=img_embed,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )  # (1,1,256,256)

                up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (1,1,H,W)
                loss = bce(up_logits, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())

        avg_loss = running_loss / max(1, len(train_ds))
        print(f"[Epoch {epoch}] avg_loss: {avg_loss:.4f}")

        # --------- quick eval on test_ds ----------
        sam.eval()
        ious = []
        with torch.no_grad():
            for j in tqdm(range(len(test_ds)), desc=f"[Epoch {epoch}] val"):
                sample = test_ds[j]
                img_np  = sample["image_np"]
                msk_np  = sample["mask_np"]
                orig_hw = sample["orig_hw"]
                box_xyxy= resize_box_to_1024(sample["bbox_xyxy"], orig_hw)

                img_in, input_hw, orig_hw = prepare_image_for_sam(img_np)
                img_in = (img_in - sam_pixel_mean) / sam_pixel_std
                img_embed = sam.image_encoder(img_in)
                dense_pe  = sam.prompt_encoder.get_dense_pe().to(DEVICE)
                boxes_1024= torch.tensor(box_xyxy, device=DEVICE, dtype=torch.float32).view(1,1,4)
                sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=boxes_1024, masks=None)

                lowres_logits, _ = sam.mask_decoder(
                    image_embeddings=img_embed,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
                tgt_bin   = torch.from_numpy((msk_np > 127).astype(np.float32))[None,None,...].to(DEVICE)
                ious.append(iou_from_logits(up_logits, tgt_bin))

        mIoU = float(np.mean(ious)) if ious else 0.0
        print(f"[Epoch {epoch}] val mIoU = {mIoU:.4f}")

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            # decoder-only weights
            torch.save({k:v.detach().cpu() for k,v in sam.mask_decoder.state_dict().items()}, dec_best_path)
            # full model for convenience
            torch.save({k:v.detach().cpu() for k,v in sam.state_dict().items()}, full_best_path)
            print(f"  ↳ saved best: {dec_best_path.name}, {full_best_path.name} (mIoU={mIoU:.4f})")

    print("Done.")

    # %% Original cell 4 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
