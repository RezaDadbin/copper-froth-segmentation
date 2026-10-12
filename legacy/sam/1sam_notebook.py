"""Historical source export; maintained entry points live in scripts/.

Original: codes/SAM/1SAM.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    # =========================
    # Cell 1 — Setup & Config
    # =========================
    import os
    import json
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    from PIL import Image
    import cv2
    import matplotlib.pyplot as plt

    # --- SAM imports (official repo) ---
    try:
        from segment_anything import sam_model_registry
        from segment_anything.modeling import Sam            # type hints
        from segment_anything.utils.transforms import ResizeLongestSide
    except Exception as e:
        raise RuntimeError(
            "segment-anything package not found.\n"
            "Install with:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from e

    # ---- Paths (EDIT IF NEEDED) ----
    TRAIN_DIR = Path("../train1")      # folder: paired .tiff/.tif + .json
    TEST_DIR  = Path("../test1")       # folder: paired .tiff/.tif + .json

    # SAM checkpoint & variant
    CHECKPOINT_PATH = Path("./weights/sam_vit_b_01ec64.pth")  # <-- put your checkpoint here
    MODEL_TYPE = "vit_b"                                      # 'vit_b' | 'vit_l' | 'vit_h'

    # ---- Hyperparameters (minimal, safe) ----
    EPOCHS = 10
    BATCH_SIZE = 1          # IMPORTANT: keep 1 to avoid SAM decoder batch-shape pitfalls with box prompts
    LR = 1e-5
    WEIGHT_DECAY = 0.0
    NUM_WORKERS = 0        # if using Windows and you see issues, set to 0

    # ---- Device selection ----
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ---- Output directory ----
    OUT_DIR = Path("./samout/sam_finetune_out")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- SAM preprocessing (longest side = 1024) ----
    sam_resize = ResizeLongestSide(1024)

    # ---- Reproducibility ----
    def set_seed(seed: int = 1337):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(1337)

    # ---- Early checks ----
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at: {CHECKPOINT_PATH}\n"
            f"Please download a SAM checkpoint (e.g., ViT-B) and place it there."
        )

    print("Setup OK ✅")
    torch.__version__

    # %% Original cell 1 (code)
    # =========================
    # Cell 2 — Froth Dataset
    # =========================
    import tifffile as tiff

    class FrothSegmentationDataset(Dataset):
        """
        Minimal dataset for paired .tif[f] images and LabelMe .json annotations.
        Returns:
          - image_raw: H x W x 3, np.uint8
          - mask     : H x W,   np.uint8 in {0,255}
        """
        def __init__(self, root: Path, label_key: str = "froth"):
            self.root = Path(root)
            self.label_key = label_key
            self.samples = self._index_pairs()
            if not self.samples:
                raise RuntimeError(f"No paired .tif[f]/.json found in: {self.root}")

        def _index_pairs(self):
            files = [f for f in os.listdir(self.root) if f.lower().endswith((".tif", ".tiff"))]
            pairs = []
            names = set(os.listdir(self.root))
            for f in sorted(files):
                j = f.rsplit(".", 1)[0] + ".json"
                if j in names:
                    pairs.append((self.root / f, self.root / j))
            return pairs

        @staticmethod
        def _to_uint8_3ch(img: np.ndarray) -> np.ndarray:
            """
            Convert image to HWC uint8 RGB:
              - handle CHW to HWC
              - grayscale -> 3 channels
              - scale non-uint8 to [0,255] uint8
            """
            arr = np.asarray(img)
            # CHW -> HWC if needed (common in some tif stacks)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))

            # grayscale -> 3ch
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)

            # if channels not 3, convert:
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[:, :, :3]  # drop extras if present

            # ensure uint8
            if arr.dtype != np.uint8:
                m = float(arr.max()) if arr.size and arr.max() > 0 else 1.0
                arr = np.clip((arr / m) * 255.0, 0, 255).astype(np.uint8)

            return arr

        def _mask_from_labelme(self, json_path: Path, H: int, W: int) -> np.ndarray:
            """
            Build a single-channel 0/255 mask from LabelMe polygons with label==self.label_key.
            """
            mask = np.zeros((H, W), dtype=np.uint8)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            for shp in shapes:
                if shp.get("label", "") != self.label_key:
                    continue
                pts = np.array(shp.get("points", []), dtype=np.int32)
                if pts.ndim == 2 and pts.shape[0] >= 3:
                    cv2.fillPoly(mask, [pts], 255)
            return mask

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, json_path = self.samples[idx]
            # load image
            img = tiff.imread(str(img_path))
            img = self._to_uint8_3ch(img)  # HWC uint8, 3ch
            H, W = img.shape[:2]

            # build mask
            mask = self._mask_from_labelme(json_path, H, W)  # HxW uint8 {0,255}

            return img, mask

    # %% Original cell 2 (code)
    ds = FrothSegmentationDataset(TRAIN_DIR)
    img, m = ds[0]
    print(img.shape, img.dtype, m.shape, m.dtype, m.min(), m.max())
    # (H, W, 3) uint8, (H, W) uint8, 0 255

    # %% Original cell 3 (code)
    # ================================
    # Cell 3 — Preprocess & Collate
    # ================================

    def preprocess_image_for_sam(image_raw_hwc_uint8: np.ndarray):
        """
        SAM preprocessing:
          - Resize longest side to 1024 using SAM's own transform
          - Convert to CHW float32 in [0,1]
          - Zero-pad to (3, 1024, 1024) for clean batching
        Returns:
          img_1024: torch.FloatTensor, shape (3, 1024, 1024)
        """
        # resize in numpy (H' x W' x 3), keeping aspect
        img_resized = sam_resize.apply_image(image_raw_hwc_uint8)
        # to torch CHW float32 [0,1]
        img_t = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        h, w = img_t.shape[1], img_t.shape[2]
        # pad to square 1024
        out = torch.zeros((3, 1024, 1024), dtype=img_t.dtype)
        out[:, :h, :w] = img_t
        return out


    def collate_froth(batch):
        """
        Collate a list of (image_raw, mask) into a SAM-ready mini-batch.
        For each sample:
          - Build full-image box prompt in ORIGINAL coords: [0, 0, W-1, H-1]
          - Preprocess image to (3, 1024, 1024)
          - Keep mask at its ORIGINAL resolution (float {0,1}) for loss

        Returns dict:
          {
            "images_1024": FloatTensor (B, 3, 1024, 1024),
            "boxes":       FloatTensor (B, 4),
            "orig_sizes":  LongTensor  (B, 2) -> (H, W),
            "masks":       list[FloatTensor (H, W)]
          }
        """
        images_1024 = []
        boxes = []
        orig_sizes = []
        masks = []

        for (img_raw, mask_raw) in batch:
            H, W = mask_raw.shape[:2]

            # preprocess image
            img_1024 = preprocess_image_for_sam(img_raw)
            images_1024.append(img_1024)

            # full-image box in ORIGINAL coords
            box = np.array([0, 0, W - 1, H - 1], dtype=np.float32)
            boxes.append(torch.from_numpy(box))

            # record original size (H, W) for upsampling loss
            orig_sizes.append(torch.tensor([H, W], dtype=torch.int64))

            # binary mask float {0,1} in original size
            m = torch.from_numpy((mask_raw > 127).astype(np.float32))
            masks.append(m)

        images_1024 = torch.stack(images_1024, dim=0)  # (B, 3, 1024, 1024)
        boxes       = torch.stack(boxes, dim=0)        # (B, 4)
        orig_sizes  = torch.stack(orig_sizes, dim=0)   # (B, 2)

        batch_out = {
            "images_1024": images_1024,
            "boxes": boxes,
            "orig_sizes": orig_sizes,
            "masks": masks,
        }
        return batch_out

    # %% Original cell 4 (code)
    # ============================
    # Cell 4 — Loss: BCE + Dice
    # ============================

    class BCEDiceLoss(nn.Module):
        """
        Combined BCE-with-logits + soft Dice loss.
        Expects:
          - logits: Tensor (B, 1, H, W)   (raw outputs from decoder, before sigmoid)
          - target: Tensor (B, 1, H, W)   (float {0,1}, same size as logits)
        """
        def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.bce_weight = bce_weight
            self.eps = eps

        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # BCE part (numerically stable on logits)
            bce = self.bce(logits, target)

            # Soft Dice part
            probs = torch.sigmoid(logits)
            # Flatten per-sample to compute Dice over spatial dims
            # (B, 1, H, W) -> (B, H*W)
            probs_f = probs.view(probs.size(0), -1)
            target_f = target.view(target.size(0), -1)

            intersection = (probs_f * target_f).sum(dim=1)
            union = probs_f.sum(dim=1) + target_f.sum(dim=1)
            dice = 1.0 - ((2.0 * intersection + self.eps) / (union + self.eps))
            dice = dice.mean()

            return self.bce_weight * bce + (1.0 - self.bce_weight) * dice

    # instantiate once; you can tweak bce_weight if froth region is very small/large
    criterion = BCEDiceLoss(bce_weight=0.5)

    # %% Original cell 5 (code)
    # ==============================
    # Cell 5 — Load SAM & Freezing
    # ==============================

    def load_sam_for_finetune(checkpoint_path: Path, model_type: str, device: torch.device):
        """
        Load a SAM model (vit_b/l/h) with pretrained checkpoint and move to device.
        """
        model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        model.to(device)
        return model

    # 1) load model
    sam_ft: Sam = load_sam_for_finetune(CHECKPOINT_PATH, MODEL_TYPE, DEVICE)

    # 2) freeze encoders; train mask decoder only
    for p in sam_ft.image_encoder.parameters():
        p.requires_grad = True

    for p in sam_ft.prompt_encoder.parameters():
        p.requires_grad = True

    for p in sam_ft.mask_decoder.parameters():
        p.requires_grad = True

    # 3) optimizer over trainable params (mask decoder)
    optimizer = torch.optim.AdamW(
        (p for p in sam_ft.parameters() if p.requires_grad),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    print("SAM loaded.")
    print(" - image_encoder: frozen")
    print(" - prompt_encoder: frozen")
    print(" - mask_decoder: trainable")
    sam_ft

    # %% Original cell 6 (code)
    # ================================
    # Cell 6 — Helpers + DataLoaders
    # ================================

    def boxes_to_1024_space(boxes: torch.Tensor, orig_hw: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from ORIGINAL pixel coords to SAM's 1024×1024 prompt space.
        boxes:   (B, 4)  [x0, y0, x1, y1] in original image coords
        orig_hw: (B, 2)  [H, W]
        returns: (B, 4)  scaled into 1024-space
        """
        out = []
        for i in range(boxes.shape[0]):
            H, W = orig_hw[i].tolist()
            sx = 1024.0 / float(W)
            sy = 1024.0 / float(H)
            x0, y0, x1, y1 = boxes[i].tolist()
            out.append([x0 * sx, y0 * sy, x1 * sx, y1 * sy])
        return torch.tensor(out, dtype=torch.float32, device=boxes.device)


    # ================================
    # Cell 6 — PATCH: enable grads in upsample_to_orig
    # ================================

    def upsample_to_orig(logits_lowres: torch.Tensor, orig_hw: torch.Tensor) -> torch.Tensor:
        """
        Bilinearly upsample low-res logits to each sample's original H×W WITH gradients.
        logits_lowres: (B, 1, h, w)  -- requires grad
        orig_hw:       (B, 2)        -- [H, W]
        returns:       (B, 1, H, W)  -- requires grad
        """
        outs = []
        B = logits_lowres.shape[0]
        for i in range(B):
            H, W = orig_hw[i].tolist()
            up = F.interpolate(
                logits_lowres[i:i+1], size=(H, W),
                mode="bilinear", align_corners=False
            )
            outs.append(up)
        return torch.cat(outs, dim=0)



    # --- Build datasets & dataloaders ---
    train_ds = FrothSegmentationDataset(TRAIN_DIR)
    test_ds  = FrothSegmentationDataset(TEST_DIR)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin, collate_fn=collate_froth
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin, collate_fn=collate_froth
    )

    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    # quick smoke test: pull one batch and show shapes
    batch = next(iter(train_loader))
    print("images_1024:", tuple(batch["images_1024"].shape))  # (B,3,1024,1024)
    print("boxes:", tuple(batch["boxes"].shape))              # (B,4)
    print("orig_sizes:", tuple(batch["orig_sizes"].shape))    # (B,2)
    print("mask[0]:", batch["masks"][0].shape, batch["masks"][0].dtype)

    # %% Original cell 7 (markdown)
    # # ==============================
    # # Cell 7 — Train & Evaluate
    # # ==============================

    # %% Original cell 8 (code)
    # ================================
    # Cell 7A — PATCH: preprocess returns resized size; collate adds input_sizes; rebuild loaders
    # ================================

    def preprocess_image_for_sam(image_raw_hwc_uint8: np.ndarray):
        """
        SAM preprocessing with resized-size tracking:
          - Resize longest side to 1024 using SAM's transform
          - Convert to CHW float32 in [0,1]
          - Zero-pad to (3, 1024, 1024)
        Returns:
          img_1024 : (3, 1024, 1024) FloatTensor
          in_size  : (h_resized, w_resized)  BEFORE padding   <-- NEW
        """
        img_resized = sam_resize.apply_image(image_raw_hwc_uint8)        # H' x W' x 3
        img_t = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        h_resized, w_resized = img_t.shape[1], img_t.shape[2]
        out = torch.zeros((3, 1024, 1024), dtype=img_t.dtype)
        out[:, :h_resized, :w_resized] = img_t
        return out, (h_resized, w_resized)


    def collate_froth(batch):
        """
        Collate with resized-size tracking.
        Returns dict:
          images_1024: (B,3,1024,1024) float
          boxes:       (B,4) float (original coords)
          orig_sizes:  (B,2) long  (H,W)
          input_sizes: (B,2) long  (h_resized, w_resized) BEFORE padding   <-- NEW
          masks:       list[(H,W) float {0,1}]
        """
        images_1024, boxes, orig_sizes, input_sizes, masks = [], [], [], [], []

        for (img_raw, mask_raw) in batch:
            H, W = mask_raw.shape[:2]
            img_1024, (h_resized, w_resized) = preprocess_image_for_sam(img_raw)

            images_1024.append(img_1024)
            boxes.append(torch.tensor([0, 0, W - 1, H - 1], dtype=torch.float32))
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
            "input_sizes": input_sizes,   # <-- NEW
            "masks": masks,
        }

    # Rebuild loaders to use the updated collate
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin, collate_fn=collate_froth
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin, collate_fn=collate_froth
    )

    # quick sanity: one batch shapes (now includes input_sizes)
    b = next(iter(train_loader))
    print("images_1024:", tuple(b["images_1024"].shape))   # (1,3,1024,1024)
    print("boxes:", tuple(b["boxes"].shape))               # (1,4)
    print("orig_sizes:", tuple(b["orig_sizes"].shape))     # (1,2)
    print("input_sizes:", tuple(b["input_sizes"].shape))   # (1,2)
    print("mask[0]:", b["masks"][0].shape, b["masks"][0].dtype)

    # %% Original cell 9 (code)
    # ===============================================
    # Cell 7B — Train & Eval with SAM-style postprocess
    # ===============================================

    def postprocess_masks_to_original(logits_lowres: torch.Tensor,
                                      input_hw: torch.Tensor,   # (B,2): (h_resized, w_resized) BEFORE padding
                                      orig_hw: torch.Tensor     # (B,2): (H, W) original
                                      ) -> torch.Tensor:
        """
        SAM-compatible postprocess:
          1) (B,1,256,256) -> upsample to (B,1,1024,1024)
          2) crop to the resized window (h_resized, w_resized) BEFORE padding
          3) resize to original (H, W)
        Returns: (B,1,H,W) logits with gradients.
        """
        B = logits_lowres.shape[0]
        outs = []
        for i in range(B):
            h_in, w_in = map(int, input_hw[i].tolist())
            H, W = map(int, orig_hw[i].tolist())

            # 1) upsample to 1024x1024
            up_1024 = F.interpolate(
                logits_lowres[i:i+1], size=(1024, 1024),
                mode="bilinear", align_corners=False
            )
            # 2) crop away zero padding
            up_cropped = up_1024[..., :h_in, :w_in]
            # 3) resize to original size
            up_orig = F.interpolate(
                up_cropped, size=(H, W),
                mode="bilinear", align_corners=False
            )
            outs.append(up_orig)
        return torch.cat(outs, dim=0)  # (B,1,H,W)


    def train_one_epoch(model: Sam,
                        loader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        criterion: nn.Module,
                        epoch: int,
                        device: torch.device) -> float:
        model.train()  # mask decoder trainable; encoders frozen
        running_loss, n_batches = 0.0, 0

        for batch in loader:
            images_1024 = batch["images_1024"].to(device)   # (1,3,1024,1024)
            boxes       = batch["boxes"].to(device)         # (1,4) original coords
            orig_hw     = batch["orig_sizes"].to(device)    # (1,2)
            input_hw    = batch["input_sizes"].to(device)   # (1,2) resized-before-pad

            # Encode image (frozen)
            with torch.no_grad():
                img_embed = model.image_encoder(images_1024)            # (1,C,64,64)

            # Prompt encodings (full-image box mapped to 1024 space)
            boxes_1024 = boxes_to_1024_space(boxes, orig_hw)            # (1,4)
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None,
                boxes=boxes_1024.unsqueeze(1),                          # (1,1,4)
                masks=None,
            )

            # Decode a single mask
            lowres_logits, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),           # (1,C,64,64)
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )  # (1,1,256,256)

            # Proper SAM postprocess to original size
            up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (1,1,H,W)

            # Build target
            target = batch["masks"][0][None, None, ...].to(device)      # (1,1,H,W) in {0,1}

            # Loss + step
            loss = criterion(up_logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / max(1, n_batches)


    @torch.no_grad()
    def evaluate_epoch(model: Sam, loader: DataLoader, device: torch.device) -> float:
        model.eval()
        ious = []

        for batch in loader:
            images_1024 = batch["images_1024"].to(device)
            boxes       = batch["boxes"].to(device)
            orig_hw     = batch["orig_sizes"].to(device)
            input_hw    = batch["input_sizes"].to(device)

            img_embed = model.image_encoder(images_1024)
            boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None,
                boxes=boxes_1024.unsqueeze(1),
                masks=None,
            )
            lowres_logits, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

            # Proper SAM postprocess to original size
            up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (B,1,H,W)
            probs = torch.sigmoid(up_logits)  # (B,1,H,W)
            preds = (probs > 0.5).float()

            # IoU per-sample
            B = preds.shape[0]
            for i in range(B):
                gt = batch["masks"][i].to(device).float()
                p  = preds[i, 0]
                inter = (p * gt).sum()
                union = p.sum() + gt.sum() - inter + 1e-6
                iou = (inter + 1e-6) / union
                ious.append(iou.item())

        return float(np.mean(ious)) if ious else 0.0

    # %% Original cell 10 (markdown)
    # ### ==============================
    # ### Cell 8 v1 — Training Driver
    # ### ==============================

    # %% Original cell 11 (code)
    best_iou = -1.0
    best_path = OUT_DIR / f"sam_{MODEL_TYPE}_mask_decoder_best.pth"
    last_path = OUT_DIR / f"sam_{MODEL_TYPE}_mask_decoder_last.pth"

    print(f"Starting training for {EPOCHS} epoch(s)…")
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(
            sam_ft, train_loader, optimizer, criterion, epoch, DEVICE
        )
        val_iou = evaluate_epoch(sam_ft, test_loader, DEVICE)
        dt = time.time() - start

        print(f"[Epoch {epoch:02d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} | val_mIoU={val_iou:.4f} | {dt:.1f}s")

        # save "last" each epoch (optional but handy)
        torch.save(
            {"mask_decoder": sam_ft.mask_decoder.state_dict(),
             "model_type": MODEL_TYPE,
             "epoch": epoch,
             "val_mIoU": val_iou},
            last_path
        )

        # save "best" when IoU improves
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {"mask_decoder": sam_ft.mask_decoder.state_dict(),
                 "model_type": MODEL_TYPE,
                 "epoch": epoch,
                 "val_mIoU": val_iou},
                best_path
            )
            print(f"  ✅ New best mIoU: {best_iou:.4f} — saved to {best_path}")

    print("\nTraining finished.")
    print(f"Best mIoU: {best_iou:.4f} | Best checkpoint: {best_path}")
    print(f"Last epoch checkpoint: {last_path}")

    # %% Original cell 12 (markdown)
    # # ==========================================================
    # # Flexible Training Cell — choose what to fine-tune
    # # ==========================================================

    # %% Original cell 13 (code)
    import math
    from pathlib import Path

    # ---------- CONFIG ----------
    TRAIN_MODE = "full"   # "decoder_only" | "decoder+prompt" | "prompt_only" | "encoder_only" | "full"
    EPOCHS_FLEX = 10

    # Effective batch size = MICRO_BATCH * ACCUM_STEPS.
    # We keep DataLoader BATCH_SIZE=1 (stable with SAM box prompts) and simulate larger batches with accumulation.
    ACCUM_STEPS = 8                # e.g., 8 -> effective batch of 8
    USE_AMP = True                 # mixed precision on CUDA

    # Learning rates per module (override per mode below)
    LR_DECODER = 1e-5
    LR_PROMPT  = 5e-6
    LR_ENCODER = 1e-6
    WEIGHT_DECAY = 0.0
    GRAD_CLIP_NORM = 1.0           # set None to disable

    # Optional: warm start the mask decoder from your 15 MB file
    LOAD_DECODER_FROM = OUT_DIR / f"sam_{MODEL_TYPE}_mask_decoder_best.pth"  # or None

    # ---------- MODE PRESETS ----------
    mode = TRAIN_MODE.lower()
    if mode == "decoder_only":
        pass  # defaults are fine
    elif mode == "decoder+prompt":
        LR_PROMPT = max(LR_PROMPT, 5e-6)
    elif mode == "prompt_only":
        LR_DECODER = 0.0
        LR_ENCODER = 0.0
        LR_PROMPT  = max(LR_PROMPT, 5e-6)
    elif mode == "encoder_only":
        LR_DECODER = 0.0
        LR_PROMPT  = 0.0
        LR_ENCODER = max(LR_ENCODER, 1e-6)
    elif mode == "full":
        LR_DECODER = max(LR_DECODER, 1e-5)
        LR_PROMPT  = max(LR_PROMPT, 5e-6)
        LR_ENCODER = max(LR_ENCODER, 1e-6)
    else:
        raise ValueError(f"Unknown TRAIN_MODE: {TRAIN_MODE}")

    print(f"TRAIN_MODE = {TRAIN_MODE}")
    print(f"LRs -> decoder: {LR_DECODER} | prompt: {LR_PROMPT} | encoder: {LR_ENCODER}")

    # ---------- Load fresh SAM ----------
    sam_train = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH)).to(DEVICE)
    sam_train.train()

    # Warm-start decoder if requested
    if LOAD_DECODER_FROM and Path(LOAD_DECODER_FROM).exists():
        _state = torch.load(LOAD_DECODER_FROM, map_location=DEVICE)
        sam_train.mask_decoder.load_state_dict(_state["mask_decoder"], strict=True)
        print(f"Warm-started decoder from: {LOAD_DECODER_FROM}")

    # ---------- Set trainable params according to mode ----------
    def set_trainables(m: Sam, mode: str):
        mode = mode.lower()
        # default: freeze all
        for p in m.parameters():
            p.requires_grad = False

        if mode in {"decoder_only", "decoder+prompt", "full"}:
            for p in m.mask_decoder.parameters():
                p.requires_grad = True

        if mode in {"prompt_only", "decoder+prompt", "full"}:
            for p in m.prompt_encoder.parameters():
                p.requires_grad = True

        if mode in {"encoder_only", "full"}:
            for p in m.image_encoder.parameters():
                p.requires_grad = True

    set_trainables(sam_train, TRAIN_MODE)

    # ---------- Build optimizer with param groups ----------
    param_groups = []
    if LR_DECODER > 0:
        param_groups.append({"params": [p for p in sam_train.mask_decoder.parameters() if p.requires_grad],
                             "lr": LR_DECODER, "weight_decay": WEIGHT_DECAY})
    if LR_PROMPT > 0:
        param_groups.append({"params": [p for p in sam_train.prompt_encoder.parameters() if p.requires_grad],
                             "lr": LR_PROMPT, "weight_decay": WEIGHT_DECAY})
    if LR_ENCODER > 0:
        param_groups.append({"params": [p for p in sam_train.image_encoder.parameters() if p.requires_grad],
                             "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY})

    # Edge case: if a group is empty (e.g., mode froze it), optimizer still needs at least one group
    if not any(len(g["params"]) for g in param_groups):
        raise RuntimeError("No trainable parameters for selected TRAIN_MODE and LRs.")

    optimizer = torch.optim.AdamW(param_groups)

    # ---------- Utility: count trainable params ----------
    def count_trainable_params(m: nn.Module):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"Trainable params: {count_trainable_params(sam_train):,}")

    # ---------- Proper SAM postprocess (256 -> 1024 -> crop -> original) ----------
    def postprocess_masks_to_original(logits_lowres: torch.Tensor,
                                      input_hw: torch.Tensor,   # (B,2) h_resized,w_resized
                                      orig_hw: torch.Tensor     # (B,2) H,W
                                      ) -> torch.Tensor:
        outs = []
        for i in range(logits_lowres.shape[0]):
            h_in, w_in = map(int, input_hw[i].tolist())
            H, W = map(int, orig_hw[i].tolist())
            up_1024 = F.interpolate(logits_lowres[i:i+1], size=(1024, 1024),
                                    mode="bilinear", align_corners=False)
            up_cropped = up_1024[..., :h_in, :w_in]
            up_orig = F.interpolate(up_cropped, size=(H, W),
                                    mode="bilinear", align_corners=False)
            outs.append(up_orig)
        return torch.cat(outs, dim=0)

    # ---------- Loss ----------
    criterion = criterion  # from Cell 4 (BCEDiceLoss), already defined

    # ---------- AMP scaler ----------
    use_amp = USE_AMP and (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---------- Flexible train/eval loops (with AMP + accumulation) ----------
    def train_one_epoch_flex(model: Sam, loader, optimizer, epoch) -> float:
        model.train()
        running_loss = 0.0
        n_steps = 0

        # Decide if encoder should compute grads
        need_img_grads = any(p.requires_grad for p in model.image_encoder.parameters())

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            images_1024 = batch["images_1024"].to(DEVICE)  # (1,3,1024,1024)
            boxes       = batch["boxes"].to(DEVICE)        # (1,4)
            orig_hw     = batch["orig_sizes"].to(DEVICE)   # (1,2)
            input_hw    = batch.get("input_sizes", torch.full_like(orig_hw, 1024)).to(DEVICE)

            # Encode
            if need_img_grads:
                img_embed = model.image_encoder(images_1024)
            else:
                with torch.no_grad():
                    img_embed = model.image_encoder(images_1024)

            # Prompt enc
            boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
            sparse_emb, dense_emb = model.prompt_encoder(points=None,
                                                         boxes=boxes_1024.unsqueeze(1),
                                                         masks=None)

            # Decode + postprocess with AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                lowres_logits, _ = model.mask_decoder(
                    image_embeddings=img_embed,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
                target = batch["masks"][0][None, None, ...].to(DEVICE)
                loss = criterion(up_logits, target) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if ((step + 1) % ACCUM_STEPS) == 0:
                if GRAD_CLIP_NORM is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * ACCUM_STEPS
            n_steps += 1

        return running_loss / max(1, n_steps)

    @torch.no_grad()
    def evaluate_epoch_flex(model: Sam, loader) -> float:
        model.eval()
        ious = []
        for batch in loader:
            images_1024 = batch["images_1024"].to(DEVICE)
            boxes       = batch["boxes"].to(DEVICE)
            orig_hw     = batch["orig_sizes"].to(DEVICE)
            input_hw    = batch.get("input_sizes", torch.full_like(orig_hw, 1024)).to(DEVICE)

            img_embed = model.image_encoder(images_1024)
            boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
            sparse_emb, dense_emb = model.prompt_encoder(points=None,
                                                         boxes=boxes_1024.unsqueeze(1),
                                                         masks=None)
            lowres_logits, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
            probs = torch.sigmoid(up_logits)
            preds = (probs > 0.5).float()  # (B,1,H,W)

            for i in range(preds.shape[0]):
                gt = batch["masks"][i].to(DEVICE).float()
                p  = preds[i, 0]
                inter = (p * gt).sum()
                union = p.sum() + gt.sum() - inter + 1e-6
                ious.append(((inter + 1e-6) / union).item())

        return float(np.mean(ious)) if ious else 0.0

    # ---------- Train driver ----------
    best_iou = -1.0
    best_decoder_path = OUT_DIR / f"sam_{MODEL_TYPE}_{TRAIN_MODE}_decoder_best.pth"
    last_decoder_path = OUT_DIR / f"sam_{MODEL_TYPE}_{TRAIN_MODE}_decoder_last.pth"
    best_full_path    = OUT_DIR / f"sam_{MODEL_TYPE}_{TRAIN_MODE}_full_best.pth"
    last_full_path    = OUT_DIR / f"sam_{MODEL_TYPE}_{TRAIN_MODE}_full_last.pth"

    print(f"Starting flexible training for {EPOCHS_FLEX} epoch(s) with accumulation x{ACCUM_STEPS}…")
    for epoch in range(1, EPOCHS_FLEX + 1):
        t0 = time.time()
        tr_loss = train_one_epoch_flex(sam_train, train_loader, optimizer, epoch)
        val_iou = evaluate_epoch_flex(sam_train, test_loader)
        dt = time.time() - t0

        print(f"[{TRAIN_MODE}] Epoch {epoch:02d}/{EPOCHS_FLEX} | "
              f"loss={tr_loss:.4f} | val_mIoU={val_iou:.4f} | {dt:.1f}s")

        # Save decoder-only (always)
        torch.save({"mask_decoder": sam_train.mask_decoder.state_dict(),
                    "model_type": MODEL_TYPE,
                    "epoch": epoch, "val_mIoU": val_iou},
                   last_decoder_path)

        # Save full model (so you have a single big file if you want to deploy that way)
        torch.save({"model": sam_train.state_dict(),
                    "model_type": MODEL_TYPE,
                    "epoch": epoch, "val_mIoU": val_iou},
                   last_full_path)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"mask_decoder": sam_train.mask_decoder.state_dict(),
                        "model_type": MODEL_TYPE,
                        "epoch": epoch, "val_mIoU": val_iou},
                       best_decoder_path)
            torch.save({"model": sam_train.state_dict(),
                        "model_type": MODEL_TYPE,
                        "epoch": epoch, "val_mIoU": val_iou},
                       best_full_path)
            print(f"  ✅ New best mIoU {best_iou:.4f} saved:\n"
                  f"     - decoder: {best_decoder_path}\n"
                  f"     - full   : {best_full_path}")

    print("\nDone.")
    print(f"Best mIoU: {best_iou:.4f}")
    print(f"Decoder checkpoints: best={best_decoder_path}, last={last_decoder_path}")
    print(f"Full checkpoints   : best={best_full_path}, last={last_full_path}")

    # %% Original cell 14 (markdown)
    # # ================================
    # # Eval + Visualization (Base SAM + fine-tuned decoder)
    # # ================================

    # %% Original cell 15 (code)
    import torch, numpy as np, matplotlib.pyplot as plt

    # 1) Load base SAM (full ~357 MB) and fine-tuned decoder (~15 MB)
    sam_eval = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH)).to(DEVICE)
    sam_eval.eval()
    torch.set_grad_enabled(False)

    ft_path = OUT_DIR / f"sam_{MODEL_TYPE}_decoder_best.pth"  # or _last.pth
    state = torch.load(ft_path, map_location=DEVICE)
    sam_eval.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
    print(f"Loaded base SAM + fine-tuned decoder from: {ft_path}")

    # 2) SAM-style postprocess (256 -> 1024 -> crop pad -> original)
    def postprocess_masks_to_original(logits_lowres: torch.Tensor,
                                      input_hw: torch.Tensor,   # (B,2): (h_resized, w_resized)
                                      orig_hw: torch.Tensor     # (B,2): (H, W)
                                      ) -> torch.Tensor:
        outs = []
        for i in range(logits_lowres.shape[0]):
            h_in, w_in = map(int, input_hw[i].tolist())
            H, W = map(int, orig_hw[i].tolist())
            up_1024 = torch.nn.functional.interpolate(
                logits_lowres[i:i+1], size=(1024, 1024), mode="bilinear", align_corners=False
            )
            up_cropped = up_1024[..., :h_in, :w_in]
            up_orig = torch.nn.functional.interpolate(
                up_cropped, size=(H, W), mode="bilinear", align_corners=False
            )
            outs.append(up_orig)
        return torch.cat(outs, dim=0)

    # 3) Evaluate mIoU on a loader (uses input_sizes if present)
    def evaluate_loader(model, loader):
        ious = []
        for batch in loader:
            images_1024 = batch["images_1024"].to(DEVICE)
            boxes       = batch["boxes"].to(DEVICE)
            orig_hw     = batch["orig_sizes"].to(DEVICE)
            if "input_sizes" in batch:
                input_hw = batch["input_sizes"].to(DEVICE)
            else:
                # Fallback if you didn't apply the alignment patch (less precise, but runs)
                input_hw = torch.full_like(orig_hw, 1024)

            # Encode & prompt
            img_embed = model.image_encoder(images_1024)
            boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None, boxes=boxes_1024.unsqueeze(1), masks=None
            )

            # Decode one mask
            lowres_logits, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

            # Proper postprocess to original size
            up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
            probs = torch.sigmoid(up_logits)
            preds = (probs > 0.5).float()   # (B,1,H,W)

            # IoU per sample
            B = preds.shape[0]
            for i in range(B):
                gt = batch["masks"][i].to(DEVICE).float()
                p  = preds[i, 0]
                inter = (p * gt).sum()
                union = p.sum() + gt.sum() - inter + 1e-6
                ious.append(((inter + 1e-6) / union).item())

        return float(np.mean(ious)) if ious else 0.0

    # Prefer test_loader; fall back to train_loader if needed
    loader_for_eval = test_loader if 'test_loader' in globals() and len(test_ds) > 0 else train_loader
    miou = evaluate_loader(sam_eval, loader_for_eval)
    print(f"mIoU (Base encoder + your fine-tuned decoder): {miou:.4f}")

    # 4) Visualization on a single image (pull raw sample from dataset for a clean overlay)
    #    Uses preprocess_image_for_sam that returns (img_1024, (h_resized, w_resized)).
    ds = test_ds if 'test_ds' in globals() and len(test_ds) > 0 else train_ds
    idx = 0  # change to visualize another sample
    img_raw, gt_mask = ds[idx]  # HWC uint8, HxW uint8 {0,255}
    H, W = gt_mask.shape[:2]

    # Build SAM inputs
    prep = preprocess_image_for_sam(img_raw)
    if isinstance(prep, tuple):
        img_1024, (h_resized, w_resized) = prep
    else:
        # if you didn't apply the 7A patch, use naive assumption (may misalign slightly)
        img_1024 = prep
        h_resized, w_resized = 1024, 1024

    img_1024 = img_1024.unsqueeze(0).to(DEVICE)  # (1,3,1024,1024)
    box_orig = torch.tensor([[0, 0, W-1, H-1]], dtype=torch.float32, device=DEVICE)
    orig_hw  = torch.tensor([[H, W]], dtype=torch.int64, device=DEVICE)
    input_hw = torch.tensor([[h_resized, w_resized]], dtype=torch.int64, device=DEVICE)

    with torch.no_grad():
        img_embed = sam_eval.image_encoder(img_1024)
        box_1024 = boxes_to_1024_space(box_orig, orig_hw)
        sparse_emb, dense_emb = sam_eval.prompt_encoder(points=None, boxes=box_1024.unsqueeze(1), masks=None)
        lowres_logits, _ = sam_eval.mask_decoder(
            image_embeddings=img_embed,
            image_pe=sam_eval.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (1,1,H,W)
        probs = torch.sigmoid(up_logits)[0, 0].cpu().numpy()
        pred_bin = (probs > 0.7).astype(np.uint8)

    # Compute IoU for the visualized sample
    gt_bin = (gt_mask > 127).astype(np.uint8)
    inter = (pred_bin & gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - inter + 1e-6
    iou_vis = (inter + 1e-6) / union
    print(f"Visualization sample idx={idx} IoU: {iou_vis:.4f}")

    # Plot: image | ground-truth | overlay
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1); plt.title("Image"); plt.imshow(img_raw); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Ground Truth"); plt.imshow(gt_mask, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Pred (thr=0.5)"); plt.imshow(img_raw); plt.imshow(pred_bin, alpha=0.45, cmap="Reds"); plt.axis("off")
    plt.tight_layout(); plt.show()

    # %% Original cell 16 (code)
    ## attention when getting output

    # %% Original cell 17 (markdown)
    # A) If you trained decoder_only

    # %% Original cell 18 (code)
    # sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH)).to(DEVICE)
    # full = torch.load(OUT_DIR / "sam_vit_b_full_best.pth", map_location=DEVICE)
    # sam.load_state_dict(full["model"], strict=True)  # loads encoder + prompt + decoder
    # sam.eval()

    # %% Original cell 19 (markdown)
    # B) If you trained decoder+prompt / prompt_only / encoder_only / full

    # %% Original cell 20 (code)
    # sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH)).to(DEVICE)
    # full = torch.load(OUT_DIR / "sam_vit_b_full_best.pth", map_location=DEVICE)
    # sam.load_state_dict(full["model"], strict=True)  # loads encoder + prompt + decoder
    # sam.eval()

    # %% Original cell 21 (markdown)
    # # ============================================
    # # Inference (decoder_only): choose how to load
    # #  - Option A (recommended): base + small decoder_best.pth
    # #  - Option B: single full_best.pth
    # # ============================================

    # %% Original cell 22 (code)
    import torch, numpy as np, matplotlib.pyplot as plt

    # ---- choose loading strategy ----
    USE_FULL = False  # False -> base + decoder_best ; True -> full_best

    # ---- paths (edit if needed) ----
    DECODER_CKPT = OUT_DIR / "sam_vit_b_decoder_only_decoder_best.pth"
    FULL_CKPT    = OUT_DIR / "sam_vit_b_decoder_only_full_best.pth"

    # ---- postprocess: 256 -> 1024 -> crop -> original ----
    def postprocess_masks_to_original(logits_lowres: torch.Tensor,
                                      input_hw: torch.Tensor,   # (B,2): (h_resized, w_resized)
                                      orig_hw: torch.Tensor):   # (B,2): (H, W)
        outs = []
        for i in range(logits_lowres.shape[0]):
            h_in, w_in = map(int, input_hw[i].tolist())
            H, W = map(int, orig_hw[i].tolist())
            up_1024 = torch.nn.functional.interpolate(
                logits_lowres[i:i+1], size=(1024, 1024), mode="bilinear", align_corners=False
            )
            up_cropped = up_1024[..., :h_in, :w_in]
            up_orig = torch.nn.functional.interpolate(
                up_cropped, size=(H, W), mode="bilinear", align_corners=False
            )
            outs.append(up_orig)
        return torch.cat(outs, dim=0)

    # ---- load model ----
    sam_inf = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH)).to(DEVICE).eval()
    if USE_FULL:
        state = torch.load(FULL_CKPT, map_location=DEVICE)
        sam_inf.load_state_dict(state["model"], strict=True)
        print(f"Loaded full finetuned model: {FULL_CKPT}")
    else:
        state = torch.load(DECODER_CKPT, map_location=DEVICE)
        sam_inf.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
        print(f"Loaded base SAM + decoder: {DECODER_CKPT}")

    # ---- pick a sample (uses your dataset objects) ----
    ds = test_ds if 'test_ds' in globals() and len(test_ds) > 0 else train_ds
    idx = 0  # change as you like
    img_raw, gt_mask = ds[idx]
    H, W = gt_mask.shape[:2]

    # preprocess (works with either version of preprocess_image_for_sam)
    prep = preprocess_image_for_sam(img_raw)
    if isinstance(prep, tuple):
        img_1024, (h_resized, w_resized) = prep
    else:
        img_1024 = prep
        h_resized, w_resized = 1024, 1024

    img_1024 = img_1024.unsqueeze(0).to(DEVICE)
    box_orig = torch.tensor([[0, 0, W-1, H-1]], dtype=torch.float32, device=DEVICE)
    orig_hw  = torch.tensor([[H, W]], dtype=torch.int64, device=DEVICE)
    input_hw = torch.tensor([[h_resized, w_resized]], dtype=torch.int64, device=DEVICE)

    # ---- inference ----
    with torch.no_grad():
        img_embed = sam_inf.image_encoder(img_1024)
        box_1024 = boxes_to_1024_space(box_orig, orig_hw)
        sparse_emb, dense_emb = sam_inf.prompt_encoder(points=None, boxes=box_1024.unsqueeze(1), masks=None)
        lowres_logits, _ = sam_inf.mask_decoder(
            image_embeddings=img_embed,
            image_pe=sam_inf.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (1,1,H,W)
        probs = torch.sigmoid(up_logits)[0, 0].cpu().numpy()
        pred_bin = (probs > 0.5).astype(np.uint8)

    # ---- metrics & viz ----
    gt_bin = (gt_mask > 127).astype(np.uint8)
    inter = (pred_bin & gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - inter + 1e-6
    iou = (inter + 1e-6) / union
    print(f"Sample idx={idx} IoU: {iou:.4f}")

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title("Image"); plt.imshow(img_raw); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Ground Truth"); plt.imshow(gt_mask, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Pred (thr=0.5)"); plt.imshow(img_raw); plt.imshow(pred_bin, alpha=0.45, cmap="Reds"); plt.axis("off")
    plt.tight_layout(); plt.show()

    # %% Original cell 23 (markdown)
    # # ============================================
    # # Inference for other modes: decoder+prompt / prompt_only / encoder_only / full
    # # Uses the FULL checkpoint (..._full_best.pth)
    # # ============================================

    # %% Original cell 24 (code)
    import torch, numpy as np, matplotlib.pyplot as plt

    # --- choose your mode & checkpoint ---
    MODE = "full"  # "decoder+prompt" | "prompt_only" | "encoder_only" | "full"
    FULL_CKPT = OUT_DIR / f"sam_vit_b_{MODE}_full_best.pth"  # edit if your file name differs

    # --- load base SAM, then full finetuned weights ---
    sam_inf = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH)).to(DEVICE)
    state = torch.load(FULL_CKPT, map_location=DEVICE)
    sam_inf.load_state_dict(state["model"], strict=True)
    sam_inf.eval()
    print(f"Loaded FULL finetuned model for mode={MODE}: {FULL_CKPT.name}")

    # --- postprocess: 256 -> 1024 -> crop pad -> original ---
    def postprocess_masks_to_original(logits_lowres, input_hw, orig_hw):
        outs = []
        for i in range(logits_lowres.shape[0]):
            h_in, w_in = map(int, input_hw[i].tolist())
            H, W = map(int, orig_hw[i].tolist())
            up_1024 = torch.nn.functional.interpolate(logits_lowres[i:i+1], size=(1024, 1024),
                                                      mode="bilinear", align_corners=False)
            up_cropped = up_1024[..., :h_in, :w_in]
            up_orig = torch.nn.functional.interpolate(up_cropped, size=(H, W),
                                                      mode="bilinear", align_corners=False)
            outs.append(up_orig)
        return torch.cat(outs, dim=0)

    # --- pick a sample ---
    ds = test_ds if 'test_ds' in globals() and len(test_ds) > 0 else train_ds
    idx = 0  # change to visualize another
    img_raw, gt_mask = ds[idx]
    H, W = gt_mask.shape[:2]

    # preprocess (works with both versions of preprocess_image_for_sam)
    prep = preprocess_image_for_sam(img_raw)
    if isinstance(prep, tuple):
        img_1024, (h_resized, w_resized) = prep
    else:
        img_1024 = prep
        h_resized, w_resized = 1024, 1024

    img_1024 = img_1024.unsqueeze(0).to(DEVICE)
    box_orig = torch.tensor([[0, 0, W-1, H-1]], dtype=torch.float32, device=DEVICE)
    orig_hw  = torch.tensor([[H, W]], dtype=torch.int64, device=DEVICE)
    input_hw = torch.tensor([[h_resized, w_resized]], dtype=torch.int64, device=DEVICE)

    # --- inference ---
    with torch.no_grad():
        img_embed = sam_inf.image_encoder(img_1024)
        box_1024 = boxes_to_1024_space(box_orig, orig_hw)
        sparse_emb, dense_emb = sam_inf.prompt_encoder(points=None, boxes=box_1024.unsqueeze(1), masks=None)
        lowres_logits, _ = sam_inf.mask_decoder(
            image_embeddings=img_embed,
            image_pe=sam_inf.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (1,1,H,W)
        probs = torch.sigmoid(up_logits)[0, 0].cpu().numpy()
        pred_bin = (probs > 0.5).astype(np.uint8)

    # --- IoU & viz ---
    gt_bin = (gt_mask > 127).astype(np.uint8)
    inter = (pred_bin & gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - inter + 1e-6
    iou = (inter + 1e-6) / union
    print(f"[{MODE}] sample idx={idx} IoU: {iou:.4f}")

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title("Image"); plt.imshow(img_raw); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Ground Truth"); plt.imshow(gt_mask, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Pred (thr=0.5)"); plt.imshow(img_raw); plt.imshow(pred_bin, alpha=0.45, cmap="Reds"); plt.axis("off")
    plt.tight_layout(); plt.show()

    # %% Original cell 25 (code)
    # ================================
    # AMG with FINETUNED SAM (per-froth contours)
    # ================================
    import os, numpy as np, cv2, torch, matplotlib.pyplot as plt
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    # --- choose how you trained ---
    USE_FULL = False  # False -> base SAM + your finetuned decoder; True -> single full finetuned file
    MODEL_TYPE = "vit_b"
    CHECKPOINT_PATH = "./weights/sam_vit_b_01ec64.pth"   # base SAM checkpoint

    # update these names to match your saved files
    DECODER_CKPT = OUT_DIR / "sam_vit_b_decoder_only_decoder_best.pth"
    FULL_CKPT    = OUT_DIR / "sam_vit_b_full_best.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load finetuned model
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH)).to(DEVICE).eval()
    if USE_FULL:
        state = torch.load(FULL_CKPT, map_location=DEVICE)
        sam.load_state_dict(state["model"], strict=True)
        print(f"Loaded FULL finetuned: {os.path.basename(FULL_CKPT)}")
    else:
        state = torch.load(DECODER_CKPT, map_location=DEVICE)
        sam.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
        print(f"Loaded base SAM + finetuned DECODER: {os.path.basename(DECODER_CKPT)}")

    # 2) AMG generator (tune thresholds to match your earlier visuals)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,            # more points -> more proposals
        pred_iou_thresh=0.86,          # stricter keeps better masks
        stability_score_thresh=0.92,   # drop unstable masks
        min_mask_region_area=80        # remove tiny specks
    )

    # 3) pick a sample
    ds = test_ds if 'test_ds' in globals() and len(test_ds) > 0 else train_ds
    idx = 0
    image_raw, gt_mask = ds[idx]   # HWC uint8, HxW uint8
    masks = generator.generate(image_raw)
    print(f"AMG produced {len(masks)} masks")

    # 4) draw a contour per mask
    overlay = image_raw.copy()
    n_contours = 0
    for m in masks:
        seg = (m["segmentation"].astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 20:
            continue
        n_contours += 1
        cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 1, lineType=cv2.LINE_AA)

    print("Contours drawn:", n_contours)

    # 5) visualize
    union = np.zeros(image_raw.shape[:2], np.uint8)
    for m in masks:
        union |= (m["segmentation"].astype(np.uint8) * 255)

    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1); plt.title(f"Original #{idx}"); plt.imshow(image_raw); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Ground truth mask"); plt.imshow(gt_mask, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title(f"SAM polygons (n_masks={len(masks)}, contours={n_contours})")
    plt.imshow(overlay); plt.axis("off")
    plt.tight_layout(); plt.show()

    # %% Original cell 26 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
