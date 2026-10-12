"""Historical source export; maintained entry points live in scripts/.

Original: codes/froth_speed/froth_speed/.ipynb_checkpoints/block matching-checkpoint.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    import numpy as np
    import cv2

    def calculate_froth_speed_block_matching(image1, image2, block_size=16, search_range=8, spatial_scale=1.0, temporal_scale=1.0):
        """
        Estimates froth speed using block matching between two grayscale images.

        Parameters:
            image1 (np.ndarray): First grayscale image.
            image2 (np.ndarray): Second grayscale image.
            block_size (int): Size of blocks to compare.
            search_range (int): Range of pixels to search for matching block.
            spatial_scale (float): Spatial scale (e.g., mm/pixel)( need to camera calibration)
            temporal_scale (float): Time between frames (e.g., seconds). (camera fps) 

        Returns:
            avg_speed (float): Average froth speed.
            avg_vx (float): Average horizontal speed.
            avg_vy (float): Average vertical speed.
            motion_vectors (list): List of (x, y, dx, dy) motion vectors for visualization.
        """
        h, w = image1.shape
        vx_list = []
        vy_list = []
        motion_vectors = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = image1[y:y+block_size, x:x+block_size]
                min_sad = float('inf')
                best_dx, best_dy = 0, 0

                # Search in image2 within the search range
                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        y2 = y + dy
                        x2 = x + dx
                        if 0 <= y2 < h - block_size and 0 <= x2 < w - block_size:
                            candidate = image2[y2:y2+block_size, x2:x2+block_size]
                            sad = np.sum(np.abs(block - candidate))
                            if sad < min_sad:
                                min_sad = sad
                                best_dx, best_dy = dx, dy

                vx_list.append(best_dx)
                vy_list.append(best_dy)
                motion_vectors.append((x, y, best_dx, best_dy))

        # Convert pixel displacement to speed
        vx_array = np.array(vx_list) * spatial_scale / temporal_scale
        vy_array = np.array(vy_list) * spatial_scale / temporal_scale
        speed_array = np.sqrt(vx_array**2 + vy_array**2)

        avg_speed = np.mean(speed_array)
        avg_vx = np.mean(vx_array)
        avg_vy = np.mean(vy_array)

        return avg_speed, avg_vx, avg_vy, motion_vectors

    # %% Original cell 1 (code)
    def draw_motion_vectors(image, vectors, block_size=16, scale=2):
        """
        Draws motion vectors on the image.
    
        Parameters:
            image (np.ndarray): Grayscale or color image.
            vectors (list): List of (x, y, dx, dy) tuples.
            block_size (int): Size of each block.
            scale (float): Scale factor for visualizing displacement.
    
        Returns:
            image_with_vectors (np.ndarray): Image with arrows drawn.
        """
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for x, y, dx, dy in vectors:
            start_point = (x + block_size // 2, y + block_size // 2)
            end_point = (int(start_point[0] + dx * scale), int(start_point[1] + dy * scale))
            cv2.arrowedLine(image_color, start_point, end_point, (0, 0, 255), 1, tipLength=0.3)

        return image_color

    # %% Original cell 2 (code)
    img1 = cv2.imread('image1.tiff', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('image2.tiff', cv2.IMREAD_GRAYSCALE)
    speed, vx, vy, vectors = calculate_froth_speed_block_matching(img1, img2)
    print(f"Froth Speed: {speed:.2f} units/s, X: {vx:.2f}, Y: {vy:.2f}")
    visual = draw_motion_vectors(img1, vectors)
    cv2.imshow("Froth Motion Vectors", visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # %% Original cell 3 (markdown)
    # Stability parameters

    # %% Original cell 4 (code)


    def calculate_stability_from_block_matching(vx_list, vy_list):
        """
        Calculates motion stability from block matching vectors.

        Parameters:
            vx_list (list): List of horizontal displacements.
            vy_list (list): List of vertical displacements.

        Returns:
            stability_scores (dict): Dictionary with variance, smoothness, and directional coherence.
        """
        vx_array = np.array(vx_list)
        vy_array = np.array(vy_list)

        # 1. Variance of motion vectors
        motion_variance = np.var(vx_array) + np.var(vy_array)

        # 2. Smoothness ratio (mean / std)
        displacement = np.sqrt(vx_array**2 + vy_array**2)
        smoothness = np.mean(displacement) / (np.std(displacement) + 1e-5)

        # 3. Directional coherence
        angles = np.arctan2(vy_array, vx_array)
        direction_variance = np.var(angles)
        directional_stability = 1 / (direction_variance + 1e-5)

        return {
            "motion_variance": motion_variance,  # Spread of motion vectors,  high value = ❌ Turbulent or unstable
            "smoothness_ratio": smoothness,      # Uniformity of displacement ,  high value =✅ Smooth, stable motion
            "directional_stability": directional_stability  # Alignment of vector directions , high value = ✅ Coherent flow 
        }

    # %% Original cell 5 (code)
    # After running block matching
    speed, vx, vy, motion_vectors = calculate_froth_speed_block_matching(img1, img2)

    # Extract dx, dy from motion vectors
    vx_list = [mv[2] for mv in motion_vectors]
    vy_list = [mv[3] for mv in motion_vectors]

    stability = calculate_stability_from_block_matching(vx_list, vy_list)
    print("Stability Scores:", stability)

    # %% Original cell 6 (markdown)
    # test for 20 frames  

    # %% Original cell 7 (code)
    import cv2
    frames = []
    for i in range(20):
        path = f"image{i+1}.tiff"  # Adjust filename pattern if needed
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Warning: Could not load {path}")
            continue
        frames.append(img)

    # %% Original cell 8 (code)
    speeds = []
    vx_list_all = []
    vy_list_all = []
    for i in range(len(frames) - 1):
        speed, vx, vy, motion_vectors = calculate_froth_speed_block_matching(frames[i], frames[i+1])
        speeds.append(speed)
        # Optional: collect vx and vy for stability analysis
        vx_list = [mv[2] for mv in motion_vectors]
        vy_list = [mv[3] for mv in motion_vectors]
        vx_list_all.append(vx_list)
        vy_list_all.append(vy_list)

        print(f"Frame {i+1}-{i+2}: Speed = {speed:.2f} units/s")

    # %% Original cell 9 (code)
    speeds = []
    stability_scores = []

    for i in range(len(frames) - 1):
        img1 = frames[i]
        img2 = frames[i + 1]

        if img1 is None or img2 is None or img1.shape != img2.shape:
            print(f"⚠️ Skipping frame pair {i+1}-{i+2}")
            continue

        speed, vx, vy, motion_vectors = calculate_froth_speed_block_matching(img1, img2)

        # Extract dx, dy from motion vectors
        vx_list = [mv[2] for mv in motion_vectors]
        vy_list = [mv[3] for mv in motion_vectors]

        stability = calculate_stability_from_block_matching(vx_list, vy_list)

        speeds.append(speed)
        stability_scores.append(stability)

        print(f"✅ Frame {i+1}-{i+2}: Speed = {speed:.2f} units/s")
        print(f"   Stability → Variance: {stability['motion_variance']:.2f}, "
              f"Smoothness: {stability['smoothness_ratio']:.2f}, "
              f"Direction: {stability['directional_stability']:.2f}")

    # %% Original cell 10 (markdown)
    # plot Froth Speed Over Time and  Directional Stability Over Time

    # %% Original cell 11 (code)
    import matplotlib.pyplot as plt

    # Plot speed
    plt.figure(figsize=(10, 4))
    plt.plot(speeds, label="Froth Speed", marker='o')
    plt.title("Froth Speed Over Time")
    plt.xlabel("Frame Pair Index")
    plt.ylabel("Speed (units/s)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot directional stability
    directional_stability = [s['directional_stability'] for s in stability_scores]

    plt.figure(figsize=(10, 4))
    plt.plot(directional_stability, label="Directional Stability", marker='s', color='green')
    plt.title("Directional Stability Over Time")
    plt.xlabel("Frame Pair Index")
    plt.ylabel("Stability Score")
    plt.grid(True)
    plt.legend()
    plt.show()

    # %% Original cell 12 (code)
    import cv2
    import numpy as np

    # Load frames
    frames = []
    for i in range(20):
        path = f"image{i+1}.tiff"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Frame {path} could not be loaded.")
            continue
        frames.append(img)

    # Test parameters
    block_sizes = [8, 16, 32]
    search_ranges = [4, 8, 12]

    # Run tests for each parameter combination
    for bs in block_sizes:
        for sr in search_ranges:
            print(f"\n🧪 Testing block_size={bs}, search_range={sr}")
            speeds = []
            directional_stabilities = []

            for i in range(len(frames) - 1):
                img1 = frames[i]
                img2 = frames[i + 1]

                if img1.shape != img2.shape:
                    print(f"⚠️ Frame pair {i+1}-{i+21} has mismatched sizes.")
                    continue

                # Calculate speed and motion vectors
                speed, vx, vy, motion_vectors = calculate_froth_speed_block_matching(
                    img1, img2,
                    block_size=bs,
                    search_range=sr,
                    spatial_scale=0.1,
                    temporal_scale=0.04
                )

                # Calculate stability
                vx_list = [mv[2] for mv in motion_vectors]
                vy_list = [mv[3] for mv in motion_vectors]
                stability = calculate_stability_from_block_matching(vx_list, vy_list)

                speeds.append(speed)
                directional_stabilities.append(stability['directional_stability'])

            # Show average results
            avg_speed = np.mean(speeds)
            avg_stability = np.mean(directional_stabilities)
            print(f"📊 Average Speed: {avg_speed:.2f} units/s")
            print(f"📊 Average Directional Stability: {avg_stability:.2f}")

    # %% Original cell 13 (code)

    # %% Original cell 14 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
