"""Historical source export; maintained entry points live in scripts/.

Original: codes/HQSAM/2HQSAM.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    # =========================
    # Cell 1 — Setup & Config
    # =========================
    import sys, os, json, math, time
    from pathlib import Path

    # --- minimal deps (install once if needed) ---
    try:
        import segment_anything_hq  # noqa: F401
    except ImportError:
        pass  # IPython/setup disabled: %pip install -q segment-anything-hq

    # base SAM's resize helper (optional but convenient)
    try:
        from segment_anything.utils.transforms import ResizeLongestSide
        _HAS_SAM_RESIZE = True
    except Exception:
        _HAS_SAM_RESIZE = False

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import tifffile as tiff
    import cv2
    import matplotlib.pyplot as plt

    # --- HQ-SAM registry ---
    from segment_anything_hq import sam_model_registry as hq_sam_model_registry

    # ---------- paths ----------
    TRAIN_DIR = Path("../train1")          # your training folder
    TEST_DIR  = Path("../test1")           # your test folder
    OUT_DIR   = Path("./samout/hqsam_finetune_out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- HQ-SAM model config ----------
    HQ_MODEL_TYPE      = "vit_b"                          # "vit_b" | "vit_l" | "vit_h" (match your checkpoint)
    HQ_CHECKPOINT_PATH = "./weights/sam_hq_vit_b.pth"     # <-- put your HQ-SAM checkpoint here

    # ---------- device ----------
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu"))
    print("Device:", DEVICE)

    # ---------- training hyperparams (we can tweak later) ----------
    EPOCHS       = 10
    BATCH_SIZE   = 1         # keep 1 for SAM/HQ-SAM box prompts; use grad accumulation for bigger effective batch
    NUM_WORKERS  = 0
    LR           = 1e-5      # good starting LR when training decoder-only
    WEIGHT_DECAY = 0.0

    # ---------- 1024-side resize helper ----------
    if _HAS_SAM_RESIZE:
        sam_resize = ResizeLongestSide(1024)
        def apply_resize_1024(img_hwc_uint8: np.ndarray) -> np.ndarray:
            return sam_resize.apply_image(img_hwc_uint8)
    else:
        # Safe fallback if base SAM isn't installed: resize longest side to 1024 with OpenCV
        def apply_resize_1024(img_hwc_uint8: np.ndarray) -> np.ndarray:
            h, w = img_hwc_uint8.shape[:2]
            if max(h, w) == 1024:
                return img_hwc_uint8
            scale = 1024.0 / float(max(h, w))
            nh, nw = int(round(h*scale)), int(round(w*scale))
            return cv2.resize(img_hwc_uint8, (nw, nh), interpolation=cv2.INTER_AREA)

    # ---------- loader for HQ-SAM ----------
    def load_hqsam(checkpoint_path=HQ_CHECKPOINT_PATH, model_type=HQ_MODEL_TYPE, device=DEVICE):
        if not os.path.exists(checkpoint_path):
            print(f"[WARN] HQ-SAM checkpoint not found at: {checkpoint_path}")
            print("Place your sam_hq checkpoint there before running training/inference.")
        model = hq_sam_model_registry[model_type](checkpoint=checkpoint_path)
        model.to(device=device).eval()
        return model

    print("HQ-SAM setup ready. Paths:")
    print(" - TRAIN_DIR:", TRAIN_DIR.resolve())
    print(" - TEST_DIR :", TEST_DIR.resolve())
    print(" - OUT_DIR  :", OUT_DIR.resolve())
    print(" - HQ checkpoint:", HQ_CHECKPOINT_PATH)

    # %% Original cell 1 (code)
    # ================================
    # Cell 2 — Dataset (TIFF + LabelMe)
    # ================================
    class FrothSegmentationDataset(Dataset):
        """
        Expects a folder containing pairs:
          - image:  *.tiff / *.tif
          - label:  same basename with .json (LabelMe format)
        Returns:
          image_raw: HWC uint8, 3-channel
          mask     : HxW  uint8 in {0,255}
        """
        IMG_EXTS = (".tiff", ".tif")

        def __init__(self, data_dir: Path):
            self.data_dir = Path(data_dir)
            self.samples = self._find_pairs()
            if len(self.samples) == 0:
                print(f"[WARN] No (image,json) pairs found in {self.data_dir}")

        def _find_pairs(self):
            pairs = []
            if not self.data_dir.exists():
                return pairs
            files = {f.name for f in self.data_dir.iterdir() if f.is_file()}
            for f in files:
                lf = f.lower()
                if lf.endswith(self.IMG_EXTS):
                    base = f[: -len(lf.split('.')[-1]) - 1]  # remove extension reliably
                    # build expected json names for both .tif and .tiff
                    json_name = base + ".json"
                    if json_name in files:
                        pairs.append((f, json_name))
            pairs.sort()
            return pairs

        def __len__(self):
            return len(self.samples)

        def _load_raw_image(self, image_path: Path) -> np.ndarray:
            """
            Load TIFF as HWC uint8 (3-channel). Converts CHW->HWC, grayscale->RGB,
            and safely scales >8-bit into [0,255].
            """
            arr = tiff.imread(str(image_path))
            arr = np.array(arr)

            # If shape is (C,H,W), transpose to (H,W,C)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[2]:
                arr = np.transpose(arr, (1, 2, 0))

            # If grayscale, stack to 3 channels
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)

            # If >8-bit, scale to uint8 preserving relative intensities
            if arr.dtype != np.uint8:
                maxv = float(arr.max()) if arr.max() > 0 else 1.0
                arr = np.clip((arr / maxv) * 255.0, 0, 255).astype(np.uint8)

            # If has 4th alpha channel, drop it
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            return arr

        def __getitem__(self, idx: int):
            img_name, json_name = self.samples[idx]
            img_path  = self.data_dir / img_name
            json_path = self.data_dir / json_name

            # Load image (HWC uint8, 3-channel)
            try:
                image_raw = self._load_raw_image(img_path)
                H, W = image_raw.shape[:2]
            except Exception as e:
                raise RuntimeError(f"Failed to load image {img_path}: {e}")

            # Build mask from LabelMe polygons
            mask = np.zeros((H, W), dtype=np.uint8)
            try:
                with open(json_path, "r") as f:
                    ann = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to read JSON {json_path}: {e}")

            got_any = False
            shapes = ann.get("shapes", [])
            for shp in shapes:
                label = shp.get("label", "")
                pts = np.array(shp.get("points", []), dtype=np.int32)
                if pts.size == 0:
                    continue
                # strictly use 'froth' label; if your JSONs vary, you can relax this
                if label == "froth":
                    cv2.fillPoly(mask, [pts], 255)
                    got_any = True

            # (optional) if no 'froth' labeled polygons, fill all polygons
            if not got_any and len(shapes) > 0:
                for shp in shapes:
                    pts = np.array(shp.get("points", []), dtype=np.int32)
                    if pts.size == 0:
                        continue
                    cv2.fillPoly(mask, [pts], 255)

            return image_raw, mask  # HWC uint8, HxW uint8

        def visualize(self, idx: int):
            """Quick visual check for a sample."""
            image_raw, mask = self[idx]
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1); plt.imshow(image_raw); plt.title(f"Image #{idx}"); plt.axis("off")
            plt.subplot(1, 2, 2); plt.imshow(mask, cmap="gray"); plt.title("GT Mask"); plt.axis("off")
            plt.tight_layout(); plt.show()


    # ---- instantiate datasets (no DataLoaders yet) ----
    train_ds = FrothSegmentationDataset(TRAIN_DIR)
    test_ds  = FrothSegmentationDataset(TEST_DIR)

    print(f"train_ds: {len(train_ds)} samples")
    print(f"test_ds : {len(test_ds)} samples")

    # Optional: quick visual sanity check if you want
    if len(train_ds) > 0:
        train_ds.visualize(0)

    # %% Original cell 2 (code)
    # ============================================
    # Cell 3 — Preprocess, Collate, DataLoaders
    # ============================================

    def preprocess_image_for_hqsam(image_raw_hwc_uint8: np.ndarray):
        """
        Resize longest side to 1024, convert to CHW float32 in [0,1], and pad to (3,1024,1024).
        Returns:
          img_1024 : torch.FloatTensor (3,1024,1024)
          in_size  : (h_resized, w_resized) BEFORE padding (for perfect mask alignment)
        """
        img_resized = apply_resize_1024(image_raw_hwc_uint8)       # H' x W' x 3, uint8
        img_t = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        h_resized, w_resized = img_t.shape[1], img_t.shape[2]
        out = torch.zeros((3, 1024, 1024), dtype=img_t.dtype)
        out[:, :h_resized, :w_resized] = img_t
        return out, (h_resized, w_resized)


    def boxes_to_1024_space(boxes: torch.Tensor, orig_hw: torch.Tensor) -> torch.Tensor:
        """
        Scale [x0,y0,x1,y1] from original pixel space to HQ-SAM's 1024×1024 prompt space.
        boxes:   (B,4) float  in original coords
        orig_hw: (B,2) long   (H, W)
        returns: (B,4) float in 1024-space
        """
        out = []
        for i in range(boxes.shape[0]):
            H, W = orig_hw[i].tolist()
            sx = 1024.0 / float(W)
            sy = 1024.0 / float(H)
            x0, y0, x1, y1 = boxes[i].tolist()
            out.append([x0 * sx, y0 * sy, x1 * sx, y1 * sy])
        return torch.tensor(out, dtype=torch.float32, device=boxes.device)


    def collate_hq(batch):
        """
        Batch = list of (image_raw HWC uint8, mask HxW uint8)
        Returns dict usable by HQ-SAM:
          images_1024: (B,3,1024,1024) float
          boxes      : (B,4) float (original coords; full-image box here)
          orig_sizes : (B,2) long  (H, W)
          input_sizes: (B,2) long  (h_resized, w_resized) BEFORE padding
          masks      : list[(H,W) float {0,1}]
        """
        images_1024, boxes, orig_sizes, input_sizes, masks = [], [], [], [], []
        for (img_raw, mask_raw) in batch:
            H, W = mask_raw.shape[:2]
            img_1024, (h_resized, w_resized) = preprocess_image_for_hqsam(img_raw)

            images_1024.append(img_1024)
            boxes.append(torch.tensor([0, 0, W - 1, H - 1], dtype=torch.float32))  # full-image box
            orig_sizes.append(torch.tensor([H, W], dtype=torch.int64))
            input_sizes.append(torch.tensor([h_resized, w_resized], dtype=torch.int64))
            masks.append(torch.from_numpy((mask_raw > 127).astype(np.float32)))

        images_1024 = torch.stack(images_1024, dim=0)
        boxes       = torch.stack(boxes, dim=0)
        orig_sizes  = torch.stack(orig_sizes, dim=0)
        input_sizes = torch.stack(input_sizes, dim=0)

        return {
            "images_1024": images_1024,
            "boxes": boxes,
            "orig_sizes": orig_sizes,
            "input_sizes": input_sizes,
            "masks": masks,
        }


    # ---- DataLoaders (keep BATCH_SIZE=1; use grad accumulation later for bigger effective batch) ----
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin, collate_fn=collate_hq
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin, collate_fn=collate_hq
    )

    # quick sanity check
    b = next(iter(train_loader)) if len(train_ds) > 0 else None
    if b is not None:
        print("images_1024:", tuple(b["images_1024"].shape))  # (B,3,1024,1024)
        print("boxes      :", tuple(b["boxes"].shape))        # (B,4)
        print("orig_sizes :", tuple(b["orig_sizes"].shape))   # (B,2)
        print("input_sizes:", tuple(b["input_sizes"].shape))  # (B,2)
        print("mask[0]    :", b["masks"][0].shape, b["masks"][0].dtype)
    else:
        print("train_ds is empty; loaders created.")

    # %% Original cell 3 (code)
    # ================================
    # Cell 4 — Loss + Postprocess
    # ================================
    class BCEDiceLoss(nn.Module):
        """
        Binary Cross Entropy (with logits) + soft Dice loss.
        Expect logits of shape (B,1,H,W) and targets in {0,1} of shape (B,1,H,W).
        """
        def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
            super().__init__()
            self.bce_weight = bce_weight
            self.smooth = smooth
            self.bce = nn.BCEWithLogitsLoss()

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            bce = self.bce(logits, targets)

            probs = torch.sigmoid(logits)
            dims = (0, 1, 2, 3)
            inter = (probs * targets).sum(dims)
            den   = probs.sum(dims) + targets.sum(dims)
            dice = 1.0 - ((2.0 * inter + self.smooth) / (den + self.smooth))
            dice = dice.mean()

            return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


    def postprocess_masks_to_original(
        logits_lowres: torch.Tensor,   # (B,1,256,256) from HQ-SAM decoder
        input_hw: torch.Tensor,        # (B,2): (h_resized, w_resized) BEFORE padding
        orig_hw: torch.Tensor          # (B,2): (H, W) original image size
    ) -> torch.Tensor:
        """
        SAM/HQ-SAM postprocess with gradients:
          1) upsample low-res logits to (1024,1024)
          2) crop the padded border using (h_resized, w_resized)
          3) resize to original (H, W)
        Returns (B,1,H,W) logits.
        """
        outs = []
        B = logits_lowres.shape[0]
        for i in range(B):
            h_in, w_in = map(int, input_hw[i].tolist())
            H, W = map(int, orig_hw[i].tolist())

            up_1024 = F.interpolate(
                logits_lowres[i:i+1], size=(1024, 1024),
                mode="bilinear", align_corners=False
            )
            up_cropped = up_1024[..., :h_in, :w_in]
            up_orig = F.interpolate(
                up_cropped, size=(H, W),
                mode="bilinear", align_corners=False
            )
            outs.append(up_orig)
        return torch.cat(outs, dim=0)


    # instantiate loss for later
    criterion = BCEDiceLoss(bce_weight=0.5, smooth=1.0)
    print("Loss & postprocess ready.")

    # %% Original cell 4 (code)
    # ================================
    # Cell 5 — HQ-SAM + Train/Eval
    # ================================

    # 1) Build model
    sam_hq = load_hqsam(HQ_CHECKPOINT_PATH, HQ_MODEL_TYPE, DEVICE)  # from Cell 1
    print("HQ-SAM loaded.")

    # 2) Trainable modules (decoder-only by default)
    for p in sam_hq.parameters():
        p.requires_grad = False
    for p in sam_hq.mask_decoder.parameters():
        p.requires_grad = True

    # (Optional) sanity: count trainable params
    def _count_trainable(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable params (decoder-only): {_count_trainable(sam_hq):,}")

    # 3) Optimizer (AdamW is fine here)
    optimizer = torch.optim.AdamW(
        [p for p in sam_hq.mask_decoder.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    # 4) Train/Eval loops
    # ===== Cell 5B — HQ train/eval (uses encode_image_hq + passes required decoder args) =====
    import torch
    import torch.nn.functional as F

    def train_one_epoch_hq(model: nn.Module,
                           loader: DataLoader,
                           optimizer: torch.optim.Optimizer,
                           criterion: nn.Module,
                           device: torch.device,
                           hq_token_only: bool = True) -> float:
        """
        Train one epoch (decoder-only fine-tuning by default).
        Expects collate keys: images_1024, boxes, orig_sizes, input_sizes, masks
        """
        model.train()  # only decoder has grads (as set earlier)
        running_loss, n_batches = 0.0, 0

        for batch in loader:
            images_1024 = batch["images_1024"].to(device)   # (B,3,1024,1024)
            boxes       = batch["boxes"].to(device)         # (B,4) in original coords
            orig_hw     = batch["orig_sizes"].to(device)    # (B,2) (H,W)
            input_hw    = batch["input_sizes"].to(device)   # (B,2) (h_resized,w_resized)
            # targets: list[(H,W)] -> (B,1,H,W) float
            targets = torch.stack(batch["masks"], dim=0).unsqueeze(1).to(device)  # (B,1,H,W)

            # 1) HQ encoder -> img_embed + interm_embeddings
            with torch.no_grad():  # encoder frozen in decoder-only setup
                img_embed, interm_embeddings = encode_image_hq(model, images_1024)  # <-- key

            # 2) prompts (map boxes to 1024 space)
            boxes_1024 = boxes_to_1024_space(boxes, orig_hw)                        # (B,4)
            sparse_emb, dense_emb = model.prompt_encoder(points=None,
                                                         boxes=boxes_1024.unsqueeze(1),
                                                         masks=None)

            # 3) HQ decoder call — MUST pass the two extra args
            lowres_logits, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                hq_token_only=hq_token_only,                 # <-- HQ-specific
                interm_embeddings=interm_embeddings,         # <-- HQ-specific
            )  # (B,1,256,256)

            # 4) Post-process to original (B,1,H,W) and compute loss
            up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
            loss = criterion(up_logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / max(1, n_batches)


    @torch.no_grad()
    def evaluate_epoch_hq(model: nn.Module,
                          loader: DataLoader,
                          device: torch.device,
                          hq_token_only: bool = True) -> float:
        """
        Eval over a loader — returns mean IoU (thr=0.5).
        """
        model.eval()
        ious = []

        for batch in loader:
            images_1024 = batch["images_1024"].to(device)
            boxes       = batch["boxes"].to(device)
            orig_hw     = batch["orig_sizes"].to(device)
            input_hw    = batch["input_sizes"].to(device)
            targets_l   = batch["masks"]  # list[(H,W) float {0,1}]

            # encoder (HQ)
            img_embed, interm_embeddings = encode_image_hq(model, images_1024)

            # prompts
            boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
            sparse_emb, dense_emb = model.prompt_encoder(points=None,
                                                         boxes=boxes_1024.unsqueeze(1),
                                                         masks=None)

            # decoder (pass HQ args)
            lowres_logits, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                hq_token_only=hq_token_only,
                interm_embeddings=interm_embeddings,
            )

            # postprocess + IoU
            up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (B,1,H,W)
            probs = torch.sigmoid(up_logits)
            preds = (probs > 0.5).float()

            B = preds.shape[0]
            for i in range(B):
                gt = targets_l[i].to(device).float()
                p  = preds[i, 0]
                inter = (p * gt).sum()
                union = p.sum() + gt.sum() - inter + 1e-6
                ious.append(((inter + 1e-6) / union).item())

        return float(np.mean(ious)) if ious else 0.0


    print("HQ-SAM training/eval ready (decoder-only).")
    sam_hq

    # %% Original cell 5 (code)
    # ===== Cell 5A — HQ image encoder wrapper (returns img_embed + interm_embeddings) =====
    import torch

    @torch.cuda.amp.autocast(enabled=False)
    def encode_image_hq(model, images_1024: torch.Tensor):
        """
        HQ-SAM image encoder should return (img_embed, interm_embeddings).
        This wrapper normalizes different package variants to that tuple.
        """
        enc_out = model.image_encoder(images_1024)
        if isinstance(enc_out, tuple) and len(enc_out) == 2:
            img_embed, interm_embeddings = enc_out
        else:
            # Fallbacks seen in some builds – try common attributes
            if hasattr(model, "predictor") and hasattr(model.predictor, "interm_features"):
                img_embed = model.predictor.get_image_embedding()
                interm_embeddings = model.predictor.interm_features
            elif hasattr(model.image_encoder, "interm_embeddings"):
                img_embed = enc_out
                interm_embeddings = model.image_encoder.interm_embeddings
            else:
                raise RuntimeError(
                    "HQ-SAM encoder did not expose intermediate embeddings.\n"
                    "Update 'segment-anything-hq' so image_encoder returns (embed, interm_embeddings)."
                )

        # Ensure proper device/types
        if isinstance(interm_embeddings, list):
            interm_embeddings = [t.to(img_embed.device) for t in interm_embeddings]
        elif torch.is_tensor(interm_embeddings):
            interm_embeddings = interm_embeddings.to(img_embed.device)
        else:
            raise TypeError("Unexpected type for interm_embeddings; expected list[Tensor] or Tensor.")

        return img_embed, interm_embeddings

    # %% Original cell 6 (code)
    # =========================
    # Cell 6 — Train Driver
    # =========================
    import time

    # make sure autograd is enabled (in case an eval cell disabled it)
    torch.set_grad_enabled(True)

    best_iou = -1.0
    DEC_BEST = OUT_DIR / f"sam_hq_{HQ_MODEL_TYPE}_decoder_best.pth"
    DEC_LAST = OUT_DIR / f"sam_hq_{HQ_MODEL_TYPE}_decoder_last.pth"
    FULL_BEST = OUT_DIR / f"sam_hq_{HQ_MODEL_TYPE}_full_best.pth"
    FULL_LAST = OUT_DIR / f"sam_hq_{HQ_MODEL_TYPE}_full_last.pth"

    # choose eval loader
    loader_for_eval = test_loader if 'test_ds' in globals() and len(test_ds) > 0 else train_loader
    print(f"Evaluating on: {'test' if loader_for_eval is test_loader else 'train'} set")

    print(f"Starting HQ-SAM training for {EPOCHS} epoch(s)…")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_one_epoch_hq(sam_hq, train_loader, optimizer, criterion, DEVICE)
        val_iou = evaluate_epoch_hq(sam_hq, loader_for_eval, DEVICE)
        dt = time.time() - t0

        print(f"[HQ] Epoch {epoch:02d}/{EPOCHS} | loss={tr_loss:.4f} | val mIoU={val_iou:.4f} | {dt:.1f}s")

        # --- save LAST checkpoints (decoder-only + full) ---
        torch.save(
            {"mask_decoder": sam_hq.mask_decoder.state_dict(),
             "model_type": HQ_MODEL_TYPE,
             "epoch": epoch,
             "val_mIoU": val_iou},
            DEC_LAST
        )
        torch.save(
            {"model": sam_hq.state_dict(),
             "model_type": HQ_MODEL_TYPE,
             "epoch": epoch,
             "val_mIoU": val_iou},
            FULL_LAST
        )

        # --- save BEST if improved ---
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {"mask_decoder": sam_hq.mask_decoder.state_dict(),
                 "model_type": HQ_MODEL_TYPE,
                 "epoch": epoch,
                 "val_mIoU": val_iou},
                DEC_BEST
            )
            torch.save(
                {"model": sam_hq.state_dict(),
                 "model_type": HQ_MODEL_TYPE,
                 "epoch": epoch,
                 "val_mIoU": val_iou},
                FULL_BEST
            )
            print(f"  ✅ new best mIoU {best_iou:.4f} saved to:\n"
                  f"     - {DEC_BEST.name} (decoder-only)\n"
                  f"     - {FULL_BEST.name} (full model)")

    print("\nDone.")
    print(f"Best mIoU: {best_iou:.4f}")
    print(f"Decoder ckpts: best={DEC_BEST.name}, last={DEC_LAST.name}")
    print(f"Full ckpts   : best={FULL_BEST.name}, last={FULL_LAST.name}")

    # %% Original cell 7 (markdown)
    # ##patches to fix the code

    # %% Original cell 8 (code)
    # --- Patch A: correct batched postprocess (256 -> 1024 -> crop -> orig) ---
    import torch.nn.functional as F

    def postprocess_masks_to_original(
        logits_lowres: torch.Tensor,   # (B,1,256,256)
        input_hw: torch.Tensor,        # (B,2)  (h_resized, w_resized)
        orig_hw: torch.Tensor          # (B,2)  (H, W)
    ) -> torch.Tensor:
        outs = []
        B = logits_lowres.shape[0]
        for i in range(B):
            h_in, w_in = map(int, input_hw[i].tolist())
            H,   W     = map(int, orig_hw[i].tolist())

            up_1024   = F.interpolate(logits_lowres[i:i+1], size=(1024, 1024),
                                      mode="bilinear", align_corners=False)
            up_crop   = up_1024[..., :h_in, :w_in]
            up_orig   = F.interpolate(up_crop, size=(H, W),
                                      mode="bilinear", align_corners=False)
            outs.append(up_orig)
        return torch.cat(outs, dim=0)   # (B,1,H,W)

    # %% Original cell 9 (code)
    # --- Patch B: HQ-SAM smoke test (one iter) ---
    torch.set_grad_enabled(True)
    sam_hq.train()

    batch      = next(iter(train_loader))
    images_1024 = batch["images_1024"].to(DEVICE)   # (B,3,1024,1024)
    boxes       = batch["boxes"].to(DEVICE)         # (B,4) in original coords
    orig_hw     = batch["orig_sizes"].to(DEVICE)    # (B,2)
    input_hw    = batch["input_sizes"].to(DEVICE)   # (B,2)
    targets     = torch.stack(batch["masks"], dim=0).unsqueeze(1).to(DEVICE)  # (B,1,H,W)

    # HQ encoder -> embeddings + intermediates
    with torch.no_grad():
        img_embed, interm_embeddings = encode_image_hq(sam_hq, images_1024)

    # prompts (map to 1024 space)
    boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
    sparse_emb, dense_emb = sam_hq.prompt_encoder(points=None,
                                                  boxes=boxes_1024.unsqueeze(1),
                                                  masks=None)

    # HQ decoder (must pass hq_token_only + interm_embeddings)
    lowres_logits, _ = sam_hq.mask_decoder(
        image_embeddings=img_embed,
        image_pe=sam_hq.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        hq_token_only=True,
        interm_embeddings=interm_embeddings,
    )

    # postprocess + loss + backward
    up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (B,1,H,W)
    loss = criterion(up_logits, targets)
    loss.backward()
    print("Smoke test OK — loss:", float(loss))

    # %% Original cell 10 (markdown)
    # ### =========================
    # ### Cell 7 — HQ Inference UI
    # ### =========================

    # %% Original cell 11 (code)
    import os, glob, torch, numpy as np, matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    try:
        import ipywidgets as widgets
    except Exception:
        pass  # IPython/setup disabled: %pip install -q ipywidgets
        import ipywidgets as widgets

    plt.rcParams["figure.dpi"] = 120

    # ---------- find latest checkpoints ----------
    def _latest(pattern: str):
        matches = glob.glob(os.path.join(str(OUT_DIR), pattern))
        if not matches:
            return ""
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]

    DECODER_CKPT = _latest(f"sam_hq_{HQ_MODEL_TYPE}_decoder_best.pth") or _latest(f"sam_hq_{HQ_MODEL_TYPE}_decoder_last.pth")
    FULL_CKPT    = _latest(f"sam_hq_{HQ_MODEL_TYPE}_full_best.pth")    or _latest(f"sam_hq_{HQ_MODEL_TYPE}_full_last.pth")

    # ---------- load model ----------
    def load_hq_for_infer(mode: str, decoder_ckpt: str = "", full_ckpt: str = ""):
        """
        mode: 'decoder_only' or 'full'
        - decoder_only: base HQ-SAM + small decoder ckpt
        - full:         load the single finetuned full ckpt
        """
        assert mode in {"decoder_only", "full"}
        sam = load_hqsam(HQ_CHECKPOINT_PATH, HQ_MODEL_TYPE, DEVICE)  # base HQ-SAM
        sam.eval()

        if mode == "decoder_only":
            ck = decoder_ckpt or DECODER_CKPT
            if not ck:
                raise FileNotFoundError("No decoder checkpoint found in OUT_DIR.")
            state = torch.load(ck, map_location=DEVICE)
            sam.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
            print(f"Loaded base HQ-SAM + decoder: {os.path.basename(ck)}")
        else:
            ck = full_ckpt or FULL_CKPT
            if not ck:
                raise FileNotFoundError("No full-model checkpoint found in OUT_DIR.")
            state = torch.load(ck, map_location=DEVICE)
            sam.load_state_dict(state["model"], strict=True)
            print(f"Loaded full finetuned HQ-SAM: {os.path.basename(ck)}")

        return sam

    # ---------- batched evaluation ----------
    @torch.no_grad()
    def evaluate_loader_hq(model, loader, hq_token_only=True):
        model.eval()
        ious = []
        for batch in loader:
            images_1024 = batch["images_1024"].to(DEVICE)   # (B,3,1024,1024)
            boxes       = batch["boxes"].to(DEVICE)         # (B,4) xyxy in orig coords
            orig_hw     = batch["orig_sizes"].to(DEVICE)    # (B,2)
            input_hw    = batch["input_sizes"].to(DEVICE)   # (B,2)
            targets     = [m.to(DEVICE).float() for m in batch["masks"]]

            # HQ encoder
            img_embed, interm_embeddings = encode_image_hq(model, images_1024)

            # prompts
            boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None, boxes=boxes_1024.unsqueeze(1), masks=None
            )

            # HQ decoder
            lowres_logits, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
                hq_token_only=hq_token_only,
                interm_embeddings=interm_embeddings,
            )

            # postprocess + IoU per sample
            up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (B,1,H,W)
            probs = torch.sigmoid(up_logits)
            preds = (probs > 0.5).float()
            B = preds.shape[0]
            for i in range(B):
                p  = preds[i, 0]
                gt = targets[i]
                inter = (p * gt).sum()
                union = p.sum() + gt.sum() - inter + 1e-6
                ious.append(((inter + 1e-6) / union).item())

        return float(np.mean(ious)) if ious else 0.0

    # ---------- single-sample viz ----------
    @torch.no_grad()
    def viz_sample_hq(model, ds, idx=0, thr=0.5, hq_token_only=True):
        img_raw, gt_mask = ds[idx]
        H, W = gt_mask.shape[:2]

        # preprocess
        img_1024, (h_resized, w_resized) = preprocess_image_for_hqsam(img_raw)  # from Cell 3
        img_1024 = img_1024.unsqueeze(0).to(DEVICE)
        orig_hw  = torch.tensor([[H, W]], dtype=torch.int64, device=DEVICE)
        input_hw = torch.tensor([[h_resized, w_resized]], dtype=torch.int64, device=DEVICE)
        box_orig = torch.tensor([[0, 0, W-1, H-1]], dtype=torch.float32, device=DEVICE)

        # HQ encoder
        img_embed, interm_embeddings = encode_image_hq(model, img_1024)

        # prompts
        box_1024 = boxes_to_1024_space(box_orig, orig_hw)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=box_1024.unsqueeze(1), masks=None
        )

        # HQ decoder
        lowres_logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            hq_token_only=hq_token_only,
            interm_embeddings=interm_embeddings,
        )

        # postprocess + viz
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (1,1,H,W)
        probs = torch.sigmoid(up_logits)[0, 0].cpu().numpy()
        pred_bin = (probs > thr).astype(np.uint8)

        gt_bin = (gt_mask > 127).astype(np.uint8)
        inter = (pred_bin & gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum() - inter + 1e-6
        iou = (inter + 1e-6) / union

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.title("Image"); plt.imshow(img_raw); plt.axis("off")
        plt.subplot(1,3,2); plt.title("Ground Truth"); plt.imshow(gt_mask, cmap="gray"); plt.axis("off")
        plt.subplot(1,3,3); plt.title(f"HQ Pred (thr={thr:.2f}) IoU={iou:.3f}")
        plt.imshow(img_raw); plt.imshow(pred_bin, alpha=0.45, cmap="Reds"); plt.axis("off")
        plt.tight_layout(); plt.show()

    # ---------- widgets ----------
    load_mode = widgets.ToggleButtons(
        options=[("decoder_only (base+decoder)", "decoder_only"),
                 ("full (single finetuned)", "full")],
        value="decoder_only" if DECODER_CKPT else "full",
        description="Load:",
    )
    decoder_tb = widgets.Text(value=str(DECODER_CKPT), description="decoder_ckpt:", layout=widgets.Layout(width="80%"))
    full_tb    = widgets.Text(value=str(FULL_CKPT),    description="full_ckpt:",    layout=widgets.Layout(width="80%"))

    dataset_dd = widgets.Dropdown(
        options=[("test", "test"), ("train", "train")],
        value="test" if 'test_ds' in globals() and len(test_ds) > 0 else "train",
        description="Dataset:"
    )
    def _cur_ds():
        return test_ds if dataset_dd.value == "test" else train_ds

    idx_slider = widgets.IntSlider(value=0, min=0, max=max(0, len(_cur_ds())-1), step=1, description="Index:")
    thr_slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01, description="Thresh:")
    def _update_idx(*args):
        idx_slider.max = max(0, len(_cur_ds())-1)
    dataset_dd.observe(_update_idx, names="value")

    load_btn = widgets.Button(description="Load Model", button_style="info")
    eval_btn = widgets.Button(description="Compute mIoU", button_style="primary")
    viz_btn  = widgets.Button(description="Visualize", button_style="success")
    out = widgets.Output()

    _model = {"sam": None}

    def on_load(_):
        with out:
            clear_output(wait=True)
            try:
                sam = load_hq_for_infer(load_mode.value, decoder_tb.value, full_tb.value)
                _model["sam"] = sam
                print("Model ready ✅")
            except Exception as e:
                print("⚠️ Load error:", e)

    def on_eval(_):
        with out:
            clear_output(wait=True)
            if _model["sam"] is None:
                print("Load a model first.")
                return
            loader = test_loader if dataset_dd.value == "test" else train_loader
            miou = evaluate_loader_hq(_model["sam"], loader, hq_token_only=True)
            print(f"mIoU on {dataset_dd.value}: {miou:.4f}")

    def on_viz(_):
        with out:
            clear_output(wait=True)
            if _model["sam"] is None:
                print("Load a model first.")
                return
            ds = _cur_ds()
            if len(ds) == 0:
                print("Selected dataset is empty.")
                return
            viz_sample_hq(_model["sam"], ds, idx=idx_slider.value, thr=thr_slider.value, hq_token_only=True)

    load_btn.on_click(on_load)
    eval_btn.on_click(on_eval)
    viz_btn.on_click(on_viz)

    ui = widgets.VBox([
        load_mode, decoder_tb, full_tb,
        widgets.HBox([load_btn, eval_btn, viz_btn]),
        dataset_dd, idx_slider, thr_slider
    ])
    display(ui, out)

    # %% Original cell 12 (markdown)
    # ### ================================
    # ### HQ-SAM "AMG-like" proposals + contours
    # ### ================================

    # %% Original cell 13 (code)
    import os, math, numpy as np, torch, cv2, matplotlib.pyplot as plt

    # ---- choose how to load your finetuned HQ-SAM ----
    USE_FULL = False  # False -> base HQ-SAM + small finetuned decoder; True -> single full finetuned file
    DECODER_CKPT = OUT_DIR / f"sam_hq_{HQ_MODEL_TYPE}_decoder_best.pth"
    FULL_CKPT    = OUT_DIR / f"sam_hq_{HQ_MODEL_TYPE}_full_best.pth"

    # ---- load model ----
    sam_hq_inf = load_hqsam(HQ_CHECKPOINT_PATH, HQ_MODEL_TYPE, DEVICE).eval()
    if USE_FULL:
        state = torch.load(FULL_CKPT, map_location=DEVICE)
        sam_hq_inf.load_state_dict(state["model"], strict=True)
        print(f"Loaded FULL finetuned HQ-SAM: {os.path.basename(FULL_CKPT)}")
    else:
        state = torch.load(DECODER_CKPT, map_location=DEVICE)
        sam_hq_inf.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
        print(f"Loaded base HQ-SAM + finetuned DECODER: {os.path.basename(DECODER_CKPT)}")

    # %% Original cell 14 (code)
    # ---------- simple HQ "AMG" generator ----------
    def hq_amg_generate(image_raw_hwc_uint8,
                        model,
                        points_per_side=32,
                        prob_thr=0.5,
                        pred_iou_thr=0.80,
                        min_mask_area=60,
                        nms_iou_thr=0.4,
                        hq_token_only=True):
        """
        Return a list[dict] like AMG, each dict has:
          { 'segmentation': HxW bool, 'area': int, 'score': float }
        """
        H, W = image_raw_hwc_uint8.shape[:2]
        # preprocess to 1024 canvas
        img_1024, (h_in, w_in) = preprocess_image_for_hqsam(image_raw_hwc_uint8)
        img_1024 = img_1024.unsqueeze(0).to(DEVICE)

        # encode once (HQ needs interm_embeddings)
        with torch.no_grad():
            img_embed, interm_embeddings = encode_image_hq(model, img_1024)

        # grid of points in 1024 space (restricted to valid resized window)
        xs = torch.linspace(0, max(1, w_in - 1), steps=points_per_side, device=DEVICE)
        ys = torch.linspace(0, max(1, h_in - 1), steps=points_per_side, device=DEVICE)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pts_1024 = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (N,2)

        # decode per point
        cand_masks = []
        pe = model.prompt_encoder.get_dense_pe()
        for k in range(pts_1024.shape[0]):
            pt = pts_1024[k:k+1]               # (1,2)
            pts  = pt.view(1,1,2)              # (1,1,2)
            labs = torch.ones((1,1), device=DEVICE, dtype=torch.int64)  # 1=foreground

            with torch.no_grad():
                sparse_emb, dense_emb = model.prompt_encoder(points=(pts, labs), boxes=None, masks=None)
                lowres_logits, iou_pred = model.mask_decoder(
                    image_embeddings=img_embed,
                    image_pe=pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                    hq_token_only=hq_token_only,
                    interm_embeddings=interm_embeddings,
                )  # (1,1,256,256), (1,1)

                # postprocess to original size
                input_hw = torch.tensor([[h_in, w_in]], dtype=torch.int64, device=DEVICE)
                orig_hw  = torch.tensor([[H, W]],        dtype=torch.int64, device=DEVICE)
                up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (1,1,H,W)

                prob = torch.sigmoid(up_logits)[0,0].detach().cpu().numpy()
                binm = (prob > prob_thr).astype(np.uint8)

            area = int(binm.sum())
            score = float(iou_pred[0,0].item())
            if area >= min_mask_area and score >= pred_iou_thr:
                cand_masks.append({
                    "segmentation": binm.astype(bool),
                    "area": area,
                    "score": score,
                })

        # Non-maximum suppression on masks by IoU
        cand_masks.sort(key=lambda m: m["score"], reverse=True)
        keep = []
        for m in cand_masks:
            seg = m["segmentation"]
            ok = True
            for kept in keep:
                inter = np.logical_and(seg, kept["segmentation"]).sum()
                union = seg.sum() + kept["segmentation"].sum() - inter + 1e-6
                iou = inter / union
                if iou >= nms_iou_thr:
                    ok = False
                    break
            if ok:
                keep.append(m)

        return keep  # list of dicts

    # ---------- run on one sample & draw contours ----------
    ds = test_ds if 'test_ds' in globals() and len(test_ds) > 0 else train_ds
    idx = 0  # change to inspect other images
    img_raw, gt_mask = ds[idx]

    masks = hq_amg_generate(
        img_raw, sam_hq_inf,
        points_per_side=32,     # increase for more proposals (but slower)
        prob_thr=0.5,
        pred_iou_thr=0.80,      # relax/tighten as you like
        min_mask_area=60,
        nms_iou_thr=0.32
    )
    print(f"HQ-AMG produced {len(masks)} masks")

    overlay = img_raw.copy()
    n_contours = 0
    for m in masks:
        seg = (m["segmentation"].astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 10:
            continue
        n_contours += 1
        cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 1, lineType=cv2.LINE_AA)

    print("Contours drawn:", n_contours)

    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1); plt.title(f"Original #{idx}"); plt.imshow(img_raw); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Ground truth mask"); plt.imshow(gt_mask, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title(f"HQ-SAM polygons (n_masks={len(masks)}, contours={n_contours})")
    plt.imshow(overlay); plt.axis("off")
    plt.tight_layout(); plt.show()

    # %% Original cell 15 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
