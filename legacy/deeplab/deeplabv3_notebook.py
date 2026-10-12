"""Historical source export; maintained entry points live in scripts/.

Original: codes/DEEPLAB/DeepLabV3.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    # ========================== DeepLabV3 (torchvision) — FULL SCRIPT ==========================
    # Requirements: torch, torchvision, numpy, pillow, opencv-python, matplotlib, tqdm

    import os, json, math, time, random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import cv2
    import matplotlib.pyplot as plt

    # %% Original cell 1 (code)
    # ---------- Config ----------
    DATA_ROOT_TRAIN = "../train1"   # folder with .tiff + .json pairs
    DATA_ROOT_TEST  = "../test1"    # folder with .tiff + .json pairs
    TARGET_SIZE_TRAIN = (1024, 1024)    # initial resize inside dataset before aug
    TARGET_SIZE_TEST  = (1024, 1024)
    CROP_SIZE = 513
    BATCH_TRAIN = 4
    BATCH_TEST  = 2
    EPOCHS = 100
    NUM_CLASSES = 2
    CLASS_NAMES = ['background', 'froth']
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    BASE_LR = 0.01
    POLY_POWER = 0.9
    GRAD_CLIP = 1.0
    ACCUM_STEPS = 1
    CKPT_PATH = "deeplabv3_best.pth"
    OUT_DIR = "predictions_deeplabv3"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, *args, **kwargs): return x

    # %% Original cell 2 (code)
    # ---------- Dataset (LabelMe JSON -> mask) ----------
    class CopperFrothDataset(Dataset):
        def __init__(self, root_dir, target_size=(512, 512)):
            self.root_dir = root_dir
            self.target_size = target_size
            self.image_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith('.tiff')])

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            image_filename = self.image_files[idx]
            image_path = os.path.join(self.root_dir, image_filename)
            json_path = image_path.replace('.tiff', '.json')

            # Load image
            image = np.array(Image.open(image_path).convert('RGB'))

            # Build mask from LabelMe
            with open(json_path, 'r') as f:
                data = json.load(f)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for shape in data.get('shapes', []):
                label = shape.get('label', '')
                if label != 'froth':
                    continue
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)

            # Resize to target_size
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
            mask  = cv2.resize(mask,  (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

            # Tensorize [0,1]
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask  = torch.from_numpy(mask).long()
            return image, mask

    # %% Original cell 3 (code)
    # ---------- Normalization + Augmentation wrappers ----------
    def imagenet_normalize(img_tensor):  # img in [0,1]
        for c in range(3):
            img_tensor[c] = (img_tensor[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return img_tensor

    class AugmentedDataset(Dataset):
        def __init__(self, base, size=513, scale=(0.7, 1.7), hflip_p=0.5):
            self.base, self.size, self.scale, self.hflip_p = base, size, scale, hflip_p
        def __len__(self): return len(self.base)
        def _resize_np(self, img_np, new_h, new_w):
            return cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        def _resize_mask(self, mask_np, new_h, new_w):
            return cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        def __getitem__(self, i):
            img, mask = self.base[i]  # img [3,H,W] in [0,1], mask [H,W]
            H, W = mask.shape
            s = random.uniform(*self.scale)
            new_h, new_w = max(2, int(H*s)), max(2, int(W*s))
            img_np  = img.permute(1,2,0).numpy()
            mask_np = mask.numpy().astype(np.uint8)
            img_np  = self._resize_np(img_np, new_h, new_w)
            mask_np = self._resize_mask(mask_np, new_h, new_w)
            th = tw = self.size
            pad_h, pad_w = max(th-new_h,0), max(tw-new_w,0)
            if pad_h or pad_w:
                img_np  = cv2.copyMakeBorder(img_np,0,pad_h,0,pad_w,cv2.BORDER_CONSTANT,value=0)
                mask_np = cv2.copyMakeBorder(mask_np,0,pad_h,0,pad_w,cv2.BORDER_CONSTANT,value=0)
            H2, W2 = mask_np.shape
            i0, j0 = random.randint(0, H2-th), random.randint(0, W2-tw)
            img_np, mask_np = img_np[i0:i0+th, j0:j0+tw, :], mask_np[i0:i0+th, j0:j0+tw]
            if random.random() < 0.5:
                img_np  = np.ascontiguousarray(np.flip(img_np, 1))
                mask_np = np.ascontiguousarray(np.flip(mask_np, 1))
            img_t  = torch.from_numpy(img_np).permute(2,0,1).float()
            mask_t = torch.from_numpy(mask_np).long()
            img_t = imagenet_normalize(img_t)
            return img_t, mask_t

    class EvalWrapper(Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            img, mask = self.base[i]
            img = imagenet_normalize(img)
            return img, mask

    # %% Original cell 4 (code)
    # ---------- Metrics helpers ----------
    def compute_confusion_matrix(pred, target, num_classes):
        mask = (target >= 0) & (target < num_classes)
        hist = np.bincount(
            num_classes * target[mask].cpu().numpy().astype(int) + pred[mask].cpu().numpy().astype(int),
            minlength=num_classes**2
        ).reshape(num_classes, num_classes)
        return hist

    def per_class_iou(confusion):
        tp = np.diag(confusion)
        fp = confusion.sum(axis=0) - tp
        fn = confusion.sum(axis=1) - tp
        denom = tp + fp + fn
        return np.divide(tp, denom, out=np.zeros_like(tp, dtype=float), where=denom!=0)

    def pixel_accuracy(confusion):
        correct = np.diag(confusion).sum()
        total = confusion.sum()
        return (correct / total) if total > 0 else 0.0

    def dice_from_iou(iou):
        return np.divide(2 * iou, 1 + iou, out=np.zeros_like(iou, dtype=float), where=(1+iou)!=0)

    # %% Original cell 5 (code)
    # ---------- Forward wrapper (supports torchvision dict output) ----------
    def forward_logits(model, images):
        out = model(images)
        if isinstance(out, dict) and 'out' in out:
            return out['out']
        return out

    # %% Original cell 6 (code)
    # ---------- Fast validation (mIoU + val_loss) ----------
    @torch.no_grad()
    def fast_eval_miou(model, dataloader, num_classes=2, device='cuda'):
        model.eval()
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        val_loss, n_batches = 0.0, 0
        criterion_local = nn.CrossEntropyLoss()
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            logits = forward_logits(model, images)
            val_loss += criterion_local(logits, masks).item()
            n_batches += 1
            preds = logits.argmax(dim=1)
            for p, t in zip(preds, masks):
                confusion += compute_confusion_matrix(p.detach().cpu(), t.detach().cpu(), num_classes)
        iou = per_class_iou(confusion); miou=float(np.nanmean(iou))
        return miou, iou, val_loss/max(n_batches,1), confusion

    # %% Original cell 7 (code)
    # ---------- Trainer (new torch.amp API, poly LR, ckpt, early stop) ----------
    try:
        from tqdm.auto import tqdm
    except:
        def tqdm(x, *args, **kwargs): return x

    def train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=60, poly_base_lr=0.01, poly_power=0.9,
        grad_clip_norm=1.0, accum_steps=1, num_classes=2,
        device='cuda', ckpt_path='best_model.pth',
        early_stop_patience=12
    ):
        device = torch.device(device)
        model.to(device)
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        max_iter = len(train_loader) * epochs
        curr_iter = 0
        best_miou, best_val_loss, best_epoch = -1.0, float('inf'), -1
        no_improve = 0

        for epoch in range(1, epochs+1):
            model.train()
            running_loss, running_items = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            start = time.time()
            for step, (images, masks) in enumerate(pbar, start=1):
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = forward_logits(model, images)
                    loss = criterion(logits, masks) / max(accum_steps, 1)

                if not torch.isfinite(loss):
                    print(f"[warn] non-finite loss at epoch {epoch}, step {step}: {loss.item()}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()

                if step % accum_steps == 0:
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * max(accum_steps, 1)
                running_items += 1

                lr = poly_base_lr * (1 - curr_iter / max_iter) ** poly_power
                for pg in optimizer.param_groups: pg['lr'] = lr
                curr_iter += 1
                pbar.set_postfix({"loss": f"{running_loss/running_items:.4f}", "lr": f"{lr:.5f}"})

            avg_train_loss = running_loss / max(running_items, 1)

            miou, iou_vec, val_loss, _ = fast_eval_miou(model, val_loader, num_classes=num_classes, device=device)
            print(
                f"Epoch {epoch}/{epochs} | train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | mIoU={miou:.4f} (bg={iou_vec[0]:.4f}, froth={iou_vec[1]:.4f})"
            )

            better = (miou > best_miou) or (math.isclose(miou, best_miou) and val_loss < best_val_loss)
            if better:
                best_miou, best_val_loss, best_epoch = miou, val_loss, epoch
                torch.save(model.state_dict(), ckpt_path)
                no_improve = 0
                print(f"  ↳ saved new best to {ckpt_path} (mIoU={best_miou:.4f}, val_loss={best_val_loss:.4f})")
            else:
                no_improve += 1

            if early_stop_patience is not None and no_improve >= early_stop_patience:
                print(f"Early stopping (no improvement {early_stop_patience} epochs). Best epoch {best_epoch}.")
                break

        print(f"Best @ epoch {best_epoch}: mIoU={best_miou:.4f}, val_loss={best_val_loss:.4f}")
        return {"best_epoch": best_epoch, "best_mIoU": best_miou, "best_val_loss": best_val_loss}

    # %% Original cell 8 (code)
    # ---------- Evaluate + save preds + gallery ----------
    COLORMAP = np.array([[0,0,0],[0,0,255]], dtype=np.uint8)
    RAW_MASK_DIR   = os.path.join(OUT_DIR, "raw_masks")
    COLOR_MASK_DIR = os.path.join(OUT_DIR, "color_masks")
    OVERLAY_DIR    = os.path.join(OUT_DIR, "overlays")
    os.makedirs(RAW_MASK_DIR, exist_ok=True)
    os.makedirs(COLOR_MASK_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    def colorize_mask(mask_np): return COLORMAP[mask_np]

    def denormalize(imgs_np):
        mean = np.array(IMAGENET_MEAN).reshape(3,1,1)
        std  = np.array(IMAGENET_STD ).reshape(3,1,1)
        imgs_vis = imgs_np * std + mean
        return np.clip(imgs_vis, 0, 1)

    @torch.no_grad()
    def evaluate_and_showcase(model, dataloader, device, out_dir=OUT_DIR, save_limit=None, show_n=6):
        model.eval()
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        gallery_imgs, gallery_gts, gallery_preds = [], [], []
        saved=0; shown=0; counter=0

        for images, masks in dataloader:
            images = images.to(device)
            masks  = masks.to(device)
            logits = forward_logits(model, images)
            preds = logits.argmax(dim=1)

            for p,t in zip(preds, masks):
                confusion += compute_confusion_matrix(p.detach().cpu(), t.detach().cpu(), NUM_CLASSES)

            preds_np = preds.cpu().numpy().astype(np.uint8)
            imgs_np  = images.cpu().numpy()
            imgs_vis = denormalize(imgs_np)
            masks_np = masks.cpu().numpy().astype(np.uint8)

            B = preds_np.shape[0]
            for i in range(B):
                pred_mask = preds_np[i]
                color_mask = colorize_mask(pred_mask)
                img = (imgs_vis[i].transpose(1,2,0) * 255).astype(np.uint8)

                Image.fromarray(pred_mask).save(os.path.join(RAW_MASK_DIR,   f"pred_{counter:06d}.png"))
                Image.fromarray(color_mask).save(os.path.join(COLOR_MASK_DIR, f"pred_color_{counter:06d}.png"))
                overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)
                Image.fromarray(overlay).save(os.path.join(OVERLAY_DIR, f"overlay_{counter:06d}.png"))
                counter += 1; saved += 1
                if save_limit is not None and saved >= save_limit: break

            if shown < show_n:
                take = min(show_n - shown, B)
                gallery_imgs.append(imgs_vis[:take]); gallery_gts.append(masks_np[:take]); gallery_preds.append(preds_np[:take])
                shown += take

            if save_limit is not None and saved >= save_limit: break

        iou = per_class_iou(confusion); miou=float(np.nanmean(iou)); acc=pixel_accuracy(confusion); dice=dice_from_iou(iou)
        print("\n================ Evaluation (Test Set) — DeepLabV3 ================")
        print(f"Pixel Accuracy     : {acc:.4f}")
        print(f"IoU (background)   : {iou[0]:.4f} | Dice: {dice[0]:.4f}")
        print(f"IoU (froth)        : {iou[1]:.4f} | Dice: {dice[1]:.4f}")
        print(f"mIoU               : {miou:.4f}")
        print("Confusion Matrix (rows=GT, cols=Pred):\n", confusion)
        print("===================================================================\n")

        # gallery
        if shown>0:
            imgs_show = np.concatenate(gallery_imgs, axis=0)
            gts_show  = np.concatenate(gallery_gts, axis=0)
            preds_show= np.concatenate(gallery_preds, axis=0)
            cols=3; rows=int(math.ceil(shown))
            plt.figure(figsize=(12, 4*rows))
            for idx in range(shown):
                plt.subplot(rows, cols, idx*cols+1); plt.imshow(np.clip(imgs_show[idx].transpose(1,2,0),0,1)); plt.title("Image"); plt.axis("off")
                plt.subplot(rows, cols, idx*cols+2); plt.imshow(gts_show[idx], vmin=0, vmax=1); plt.title("GT"); plt.axis("off")
                plt.subplot(rows, cols, idx*cols+3); plt.imshow(preds_show[idx], vmin=0, vmax=1); plt.title("Pred"); plt.axis("off")
            plt.tight_layout(); plt.show()

        return {"pixel_acc":acc, "iou":iou, "miou":miou, "dice":dice, "confusion":confusion}

    # %% Original cell 9 (code)
    # ---------- Build datasets/loaders ----------
    train_base = CopperFrothDataset(DATA_ROOT_TRAIN, target_size=TARGET_SIZE_TRAIN)
    test_base  = CopperFrothDataset(DATA_ROOT_TEST,  target_size=TARGET_SIZE_TEST)
    train_dataset = AugmentedDataset(train_base, size=CROP_SIZE, scale=(0.7,1.7), hflip_p=0.5)
    test_dataset  = EvalWrapper(test_base)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_TEST,  shuffle=False, num_workers=0, pin_memory=True)

    # %% Original cell 10 (code)
    # ---------- Class weights ----------
    def estimate_class_weights(dataset, num_classes=2, samples=200):
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        counts = np.zeros(num_classes, dtype=np.int64); n=0
        for _, mask in loader:
            m = mask.numpy().ravel()
            for c in range(num_classes): counts[c] += (m == c).sum()
            n += 1
            if n >= samples: break
        freqs = counts / max(counts.sum(), 1)
        weights = 1.0 / np.clip(freqs, 1e-6, 1.0)
        weights = weights / weights.mean()
        return weights.astype(np.float32)

    class_weights = estimate_class_weights(train_base, num_classes=NUM_CLASSES, samples=200)
    print("Estimated class weights:", class_weights)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

    # %% Original cell 11 (code)
    # ---------- Model (torchvision DeepLabV3) ----------
    from torchvision.models.segmentation import deeplabv3_resnet50
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead

    try:
        model = deeplabv3_resnet50(weights="DEFAULT")
    except TypeError:
        model = deeplabv3_resnet50(pretrained=True)
    model.classifier = DeepLabHead(2048, NUM_CLASSES)
    model = model.to(device)

    # %% Original cell 12 (code)
    # ---------- Optimizer ----------
    optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=1e-4, nesterov=True)

    # %% Original cell 13 (code)
    # ---------- Train ----------
    _ = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        epochs=EPOCHS, poly_base_lr=BASE_LR, poly_power=POLY_POWER,
        grad_clip_norm=GRAD_CLIP, accum_steps=ACCUM_STEPS,
        num_classes=NUM_CLASSES, device=device, ckpt_path=CKPT_PATH,
        early_stop_patience=12
    )

    # %% Original cell 14 (code)
    # ---------- Load best & Evaluate ----------
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    _ = evaluate_and_showcase(model, test_loader, device, out_dir=OUT_DIR, save_limit=None, show_n=6)
    # =========================================================================================

    # %% Original cell 15 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
