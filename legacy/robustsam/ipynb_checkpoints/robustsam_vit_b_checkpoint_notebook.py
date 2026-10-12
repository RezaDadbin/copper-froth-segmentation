"""Historical source export; maintained entry points live in scripts/.

Original: codes/ROBUSTSAM/.ipynb_checkpoints/RobustSAM_vit_b-checkpoint.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    pass  # IPython/setup disabled: pip uninstall -y segment-anything medsam segment-anything-hq

    # %% Original cell 1 (code)
    # --- SAM integration for froth segmentation (inference + eval) ---
    import os
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import cv2
    import json
    import matplotlib.pyplot as plt

    # -------- your original DataLoader (slightly extended) ----------
    import tifffile as tiff
    import torchvision.transforms as transforms

    # %% Original cell 2 (code)
    class FrothSegmentationDataLoader(Dataset):
        def __init__(self, data_dir, transform_mode='train'):
            self.data_dir = data_dir
            self.samples = self._load_filenames()
            # keep your transforms for training/visualization if needed
            self.transform = transforms.Compose([
                transforms.CenterCrop(1080),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            self.transform2 = transforms.Compose([
                transforms.CenterCrop(480),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

        def _load_filenames(self):
            samples = []
            for file in os.listdir(self.data_dir):
                json_file = None
                if file.endswith(".tiff"):
                    json_file = file.replace(".tiff", ".json")
                elif file.endswith(".tif"):
                    json_file = file.replace(".tif", ".json")
                if json_file in os.listdir(self.data_dir):
                    samples.append((file, json_file))
            return samples

        def __len__(self):
            return len(self.samples)

        def _load_raw_image(self, image_path):
            """
            Return raw numpy uint8 HWC image for SAM.
            """
            image = tiff.imread(image_path)
            # If it's multi-channel with shape (C, H, W) convert to HWC
            image = np.array(image)
            if image.ndim == 3 and image.shape[0] in [1,3,4] and image.shape[0] != image.shape[2]:
                # convert CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
            # Ensure 3-channel uint8
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            # If image is greater bit-depth (e.g., 16-bit), scale down to 0-255 carefully
            if image.dtype != np.uint8:
                # scale preserving relative intensities (clip)
                image = np.clip((image / image.max()) * 255.0, 0, 255).astype(np.uint8)
            return image

        def __getitem__(self, index):
            image_file, json_file = self.samples[index]
            image_path = os.path.join(self.data_dir, image_file)
            json_path = os.path.join(self.data_dir, json_file)

            # load raw image and mask
            try:
                image_raw = self._load_raw_image(image_path)
                height, width = image_raw.shape[:2]
            except Exception as e:
                print(f"Error loading image: {image_path}, Error: {e}")
                return None

            mask = np.zeros((height, width), dtype=np.uint8)
            try:
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
            except Exception as e:
                print(f"Error loading json {json_path}: {e}")
                return None

            for shape in annotations['shapes']:
                if shape['label'] == 'froth':
                    pts = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)

            # return both raw image (HWC uint8) and mask (H,W uint8 0/255)
            return image_raw, mask

        def visualize(self, index):
            item = self.__getitem__(index)
            if item is None:
                print("Failed to load sample")
                return
            image_raw, mask = item
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.imshow(image_raw)
            plt.title("Raw Image")
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(mask, cmap='gray')
            plt.title("GT Mask")
            plt.axis('off')
            plt.show()

    # %% Original cell 3 (code)

    # %% Original cell 4 (code)
    try:
        from medsam import medsam_model_registry, MedSamAutomaticMaskGenerator
    except Exception as e:
        print("Unable to import MedSAM. Try installing it with:")
        print("pip install git+https://github.com/bowang-lab/MedSAM.git")
        raise e

    # %% Original cell 5 (code)
    # Choose checkpoint and model_type (use small model on CPU)
    CHECKPOINT_PATH = "./weights/robustsam_checkpoint_b.pth"   # <-- put your checkpoint here
    MODEL_TYPE = "vit_b"                # smallest reasonable SAM variant

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Warning: SAM checkpoint not found at {CHECKPOINT_PATH}.")
        print("Download a checkpoint (e.g. sam_vit_b) and place it at this path before running.")
        # we still try to proceed; import will fail later when attempting to load weights.

    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    device = torch.device("mps" if use_mps else "cpu")
    print("Using device:", device)


    def load_medsam(checkpoint_path=CHECKPOINT_PATH, model_type=MODEL_TYPE, device=device):
        medsam = medsam_model_registry[model_type](checkpoint=checkpoint_path)
        medsam.to(device=device)
        medsam.eval()
        return medsam

    # Build an automatic mask generator (fastish)
    def make_automatic_mask_generator(sam_model,
                                      points_per_side=32,
                                      pred_iou_thresh=0.5,
                                      stability_score_thresh=0.95,
                                      min_mask_region_area=100):
        generator = SamAutomaticMaskGenerator(
            sam_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area
        )
        return generator

    # Utility to union automatic masks into a single binary mask (H, W) 0/255
    def union_of_masks(annotated_masks, image_shape):
        h, w = image_shape[:2]
        union_mask = np.zeros((h, w), dtype=np.uint8)
        for m in annotated_masks:
            # each m has 'segmentation' as a boolean mask (H, W) or RLE; the SAM output gives a 2D mask array
            mask = m['segmentation'].astype(np.uint8) * 255
            union_mask = np.maximum(union_mask, mask)
        return union_mask

    # Simple metric functions
    def compute_pixel_accuracy(pred_mask, gt_mask):
        # pred_mask and gt_mask expected 0/255 uint8
        total = gt_mask.size
        correct = (pred_mask == gt_mask).sum()
        return float(correct) / float(total)

    def compute_iou(pred_mask, gt_mask):
        # convert to boolean
        p = (pred_mask > 127)
        g = (gt_mask > 127)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union == 0:
            return 1.0 if inter == 0 else 0.0
        return inter / union

    def compute_dice(pred_mask, gt_mask):
        p = (pred_mask > 127)
        g = (gt_mask > 127)
        inter = 2 * np.logical_and(p, g).sum()
        denom = p.sum() + g.sum()
        if denom == 0:
            return 1.0 if inter == 0 else 0.0
        return inter / denom

    # %% Original cell 6 (code)
    def compute_f1(pred_mask, gt_mask):
        # convert masks to boolean
        p = (pred_mask > 127)
        g = (gt_mask > 127)
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, np.logical_not(g)).sum()
        fn = np.logical_and(np.logical_not(p), g).sum()
        if tp + fp + fn == 0:
            return 1.0  # both empty
        f1 = 2 * tp / (2 * tp + fp + fn)
        return f1

    def evaluate_sam_on_dataset(data_dir, checkpoint_path=CHECKPOINT_PATH, model_type=MODEL_TYPE, device="cpu"):
        dataset = FrothSegmentationDataLoader(data_dir)
        print("Dataset length:", len(dataset))

        sam = load_sam(checkpoint_path, model_type, device)
        generator = make_automatic_mask_generator(sam)

        px_accs, ious, dices, f1s = [], [], [], []

        for idx in range(len(dataset)):
            image_raw, gt_mask = dataset[idx]
            if image_raw.dtype != np.uint8:
                image_for_sam = np.clip((image_raw / image_raw.max()) * 255.0, 0, 255).astype(np.uint8)
            else:
                image_for_sam = image_raw

            sam_masks = generator.generate(image_for_sam)
            pred_union = union_of_masks(sam_masks, image_for_sam.shape)

            # remove tiny components
            min_area = 20
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((pred_union > 127).astype(np.uint8), connectivity=8)
            filtered = np.zeros_like(pred_union)
            for lab in range(1, num_labels):
                area = stats[lab, cv2.CC_STAT_AREA]
                if area >= min_area:
                    filtered[labels == lab] = 255
            pred_mask = filtered

            # compute metrics
            px_acc = compute_pixel_accuracy(pred_mask, gt_mask)
            iou = compute_iou(pred_mask, gt_mask)
            dice = compute_dice(pred_mask, gt_mask)
            f1 = compute_f1(pred_mask, gt_mask)

            px_accs.append(px_acc)
            ious.append(iou)
            dices.append(dice)
            f1s.append(f1)

            print(f"Sample {idx}: px_acc={px_acc:.4f}, IoU={iou:.4f}, Dice={dice:.4f}, F1={f1:.4f}")

        # aggregate
        mean_px = float(np.mean(px_accs))
        mean_iou = float(np.mean(ious))
        mean_dice = float(np.mean(dices))
        mean_f1 = float(np.mean(f1s))

        print("SAM Evaluation Summary:")
        print(f"Pixel accuracy: {mean_px:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Mean Dice: {mean_dice:.4f}")
        print(f"Mean F1: {mean_f1:.4f}")

        return {"pixel_accuracy": mean_px, "iou": mean_iou, "dice": mean_dice, "f1": mean_f1}

    # %% Original cell 7 (code)
    if __name__ == "__main__":
        # Evaluate SAM on the training set (this will be slow on CPU)
        results = evaluate_sam_on_dataset("t1", CHECKPOINT_PATH, MODEL_TYPE, device="cpu")
        print(results)

    # %% Original cell 8 (code)
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    # helper: union masks (each sam_mask['segmentation'] is a boolean HxW array)
    def union_of_masks(sam_masks, image_shape):
        h, w = image_shape[:2]
        union = np.zeros((h, w), dtype=np.uint8)
        for m in sam_masks:
            mask = (m['segmentation'].astype(np.uint8) * 255)
            union = np.maximum(union, mask)
        return union

    # helper: draw contours of each sam mask on a copy of the original image
    def draw_sam_polygons_on_image(image_rgb, sam_masks, contour_color=(255, 0, 0), thickness=1):
        # image_rgb is HWC uint8
        overlay = image_rgb.copy()
        # keep track of number of masks drawn
        n_drawn = 0
        for m in sam_masks:
            mask = (m['segmentation'].astype(np.uint8) * 255)
            # findContours expects single-channel uint8
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            cv2.drawContours(overlay, contours, -1, contour_color, thickness, lineType=cv2.LINE_AA)
            n_drawn += len(contours)
        return overlay, n_drawn

    # Main visualization function
    def visualize_dataset_with_sam(dataset, sam_generator, out_dir=None, max_images=None, figsize_per_row=(12, 3)):
        """
        dataset: FrothSegmentationDataLoader instance that returns (image_raw, gt_mask)
        sam_generator: SamAutomaticMaskGenerator instance
        out_dir: if provided, will save each row image into this directory
        max_images: max number of images to visualize (None -> all)
        """
        os.makedirs(out_dir, exist_ok=True) if out_dir else None
        n = len(dataset) if max_images is None else min(len(dataset), max_images)
        for i in range(n):
            image_raw, gt_mask = dataset[i]               # image_raw: HWC uint8, gt_mask: HxW uint8(0/255)
            # Ensure uint8 HWC
            assert image_raw.dtype == np.uint8, "image must be uint8 HWC"
            # Run SAM (this can be slow on CPU/MPS)
            sam_masks = sam_generator.generate(image_raw)
            # union mask (single-channel 0/255)
            union_mask = union_of_masks(sam_masks, image_raw.shape)

            # draw polygons on original
            overlay_img, n_poly = draw_sam_polygons_on_image(image_raw, sam_masks, contour_color=(255,0,0), thickness=1)

            # plot row: original | GT mask | overlay
            fig = plt.figure(figsize=figsize_per_row)
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(image_raw)
            ax1.set_title(f"Original #{i}")
            ax1.axis('off')

            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(gt_mask, cmap='gray')
            ax2.set_title("Ground truth mask")
            ax2.axis('off')

            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(overlay_img)
            # also overlay union semi-transparently for clarity:
            ax3.imshow(union_mask, alpha=0.35, cmap='Reds')
            ax3.set_title(f"SAM polygons (n_masks={len(sam_masks)}, contours={n_poly})")
            ax3.axis('off')

            plt.tight_layout()
            plt.show()

            # optionally save the row image
            if out_dir:
                save_path = os.path.join(out_dir, f"sam_vis_{i}.png")
                # convert matplotlib figure to image and save
                fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                print("Saved visualization to:", save_path)

            plt.close(fig)

    # %% Original cell 9 (code)

    train_dataset = FrothSegmentationDataLoader("t1") 

    device = "cpu"
    sam = load_sam(CHECKPOINT_PATH, MODEL_TYPE, device)
    generator = make_automatic_mask_generator(sam)


    visualize_dataset_with_sam(train_dataset, generator, out_dir="sam_vis", max_images=10)

    # %% Original cell 10 (code)

    # %% Original cell 11 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
