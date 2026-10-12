"""Historical source export; maintained entry points live in scripts/.

Original: codes/POST/froth-postprocess/eval.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    import cv2
    import numpy as np
    import json
    import os

    # Important: Make sure the name of this file matches your script's filename.
    # If your script is named 'main.py', this should be 'from main import ...'
    from your_main_script import process_image_roi_watershed

    # ------------------------------------------------------------------
    # This function loads the ground truth mask from your JSON file.
    # You will probably need to EDIT this function to match your JSON structure.
    # ------------------------------------------------------------------
    def load_ground_truth_from_json(json_path, shape):
        """
        Loads polygon data from a LabelMe-style JSON file and draws it to create a
        binary ground truth mask.

        Args:
            json_path (str): The path to the JSON file.
            shape (tuple): The (height, width) of the mask to create.

        Returns:
            np.ndarray: A binary (0/255) uint8 ground truth mask, or None if file fails.
        """
        gt_mask = np.zeros(shape, dtype=np.uint8)
    
        try:
            with open(json_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Ground truth JSON not found at {json_path}")
            return None

        # --- THIS PART IS NOW CORRECT FOR YOUR FILE ---
        # Your JSON has a top-level key "shapes" which is a list of all froth polygons.
        regions = data.get('shapes', [])
        # --- END OF CUSTOMIZED PART ---

        if not regions:
            print(f"Warning: No 'shapes' found in {json_path}. Check the JSON file.")
            return gt_mask

        for region in regions:
            try:
                # The polygon coordinates are in the 'points' key for each shape.
                points_list = region.get('points', [])
            
                if not points_list:
                    continue # Skip if this shape has no points for some reason

                # Convert the list of [x, y] points into a NumPy array that OpenCV needs.
                points_np = np.array(points_list, dtype=np.int32)
            
                # Draw the filled polygon onto our empty mask.
                cv2.fillPoly(gt_mask, [points_np], 255)

            except Exception as e:
                print(f"Warning: Could not process a shape in {json_path}. Error: {e}")
                continue
            
        return gt_mask

    def calculate_metrics(gt_mask, pred_mask):
        """
        Calculates segmentation metrics between a ground truth and predicted mask.
        Masks should be binary (0 or 255).
        """
        # Convert masks to boolean arrays for logical operations
        gt_bool = gt_mask.astype(bool)
        pred_bool = pred_mask.astype(bool)

        # Calculate metrics
        intersection = np.logical_and(gt_bool, pred_bool)
        union = np.logical_or(gt_bool, pred_bool)
    
        tp = np.sum(intersection)  # True Positives
        fp = np.sum(pred_bool) - tp # False Positives
        fn = np.sum(gt_bool) - tp   # False Negatives
        epsilon = 1e-6

        # Intersection over Union (IoU) - The most important one
        iou = tp / (np.sum(union) + epsilon)
    
        # Dice Coefficient
        dice = (2. * tp) / (np.sum(gt_bool) + np.sum(pred_bool) + epsilon)
    
        return {
            'iou': iou,
            'dice': dice,
        }

    # ------------------------------------------------------------------
    # Main Evaluation Loop
    # ------------------------------------------------------------------
    if __name__ == '__main__':
        # --- USER SETTINGS ---
        # 1. Folder with your model's segmentation map outputs
        PREDICTION_FOLDER = './output_images_viridis/' 
        # 2. Folder where you put your ground truth JSON files
        JSON_FOLDER = './ground_truth_jsons/'   
        # --- END SETTINGS ---
    
        pred_files = [f for f in sorted(os.listdir(PREDICTION_FOLDER)) if f.lower().endswith(('.png', '.jpg'))]
        all_metrics = []

        for image_name in pred_files:
            print(f"\n--- Evaluating {image_name} ---")
            pred_path = os.path.join(PREDICTION_FOLDER, image_name)
        
            # 1. Generate your final predicted mask
            final_predicted_mask, _, _, _ = process_image_roi_watershed(pred_path, visualize=False)

            # 2. Load the corresponding ground truth mask from the JSON file
            # --- THIS IS THE MODIFIED PART ---
            base_name = os.path.splitext(image_name)[0]       # e.g., "output_0_0"
            json_base_name = base_name.replace("output_", "j_") # e.g., "j_0_0"
            json_name = json_base_name + '.json'               # e.g., "j_0_0.json"
            # --- END MODIFICATION ---
        
            json_path = os.path.join(JSON_FOLDER, json_name)
            gt_mask = load_ground_truth_from_json(json_path, shape=final_predicted_mask.shape)
        
            if gt_mask is None:
                continue

            # 3. Calculate metrics and print them
            metrics = calculate_metrics(gt_mask, final_predicted_mask)
            all_metrics.append(metrics)
        
            print(f"  Intersection over Union (IoU): {metrics['iou']:.4f}")
            print(f"  Dice Score: {metrics['dice']:.4f}")

        # --- Calculate and print the average scores ---
        if all_metrics:
            avg_iou = np.mean([m['iou'] for m in all_metrics])
            avg_dice = np.mean([m['dice'] for m in all_metrics])
            print("\n========================================")
            print("    Average Scores Over All Images")
            print("========================================")
            print(f"  Average IoU:  {avg_iou:.4f}")
            print(f"  Average Dice: {avg_dice:.4f}")

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
