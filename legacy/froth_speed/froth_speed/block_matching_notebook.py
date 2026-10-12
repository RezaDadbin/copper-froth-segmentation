"""Historical source export; maintained entry points live in scripts/.

Original: codes/froth_speed/froth_speed/block matching.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)

    import os 
    os.getcwd()

    # %% Original cell 1 (code)
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    # %% Original cell 2 (code)
    def calculate_froth_speed_block_matching(image1, image2, block_size=128, search_range=90, spatial_scale=1.0, temporal_scale=1.0):
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

    # %% Original cell 3 (code)
    def calculate_froth_speed_block_matching(image1, image2, block_size=64, search_range=90, spatial_scale=1.0, temporal_scale=1.0, texture_threshold=30):
        h, w = image1.shape
        vx_list, vy_list, motion_vectors = [], [], []

        # Pre-process images (denoise + equalize contrast)
        image1 = cv2.GaussianBlur(image1, (5,5), 0)
        image2 = cv2.GaussianBlur(image2, (5,5), 0)
        image1 = cv2.equalizeHist(image1)
        image2 = cv2.equalizeHist(image2)

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = image1[y:y+block_size, x:x+block_size]

                # Skip flat/low-texture blocks
                if block.std() < texture_threshold:
                    continue

                min_sad = float('inf')
                best_dx, best_dy = 0, 0

                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        y2, x2 = y + dy, x + dx
                        if 0 <= y2 < h - block_size and 0 <= x2 < w - block_size:
                            candidate = image2[y2:y2+block_size, x2:x2+block_size]
                            sad = np.sum(np.abs(block - candidate))
                            if sad < min_sad:
                                min_sad = sad
                                best_dx, best_dy = dx, dy

                vx_list.append(best_dx)
                vy_list.append(best_dy)
                motion_vectors.append((x, y, best_dx, best_dy))

        vx_array = np.array(vx_list) * spatial_scale / temporal_scale
        vy_array = np.array(vy_list) * spatial_scale / temporal_scale
        speed_array = np.sqrt(vx_array**2 + vy_array**2)

        return np.mean(speed_array), np.mean(vx_array), np.mean(vy_array), motion_vectors

    # %% Original cell 4 (code)
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

    # %% Original cell 5 (code)
    img1 = cv2.imread('F61 (19).tiff', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('F61 (20).tiff', cv2.IMREAD_GRAYSCALE)
    # Apply a Gaussian blur before running your function
    image1_blur = cv2.GaussianBlur(img1, (15, 15), 0)
    image2_blur = cv2.GaussianBlur(img2, (15, 15), 0)


    # %% Original cell 6 (code)
    speed, vx, vy, vectors = calculate_froth_speed_block_matching(img1, img2)
    print(f"Froth Speed: {speed:.2f} units/s, X: {vx:.2f}, Y: {vy:.2f}")
    visual = draw_motion_vectors(img1, vectors)

    plt.imshow(visual, cmap="grey")
    plt.title("Froth Motion Vectors")
    plt.show()
    # cv2.imshow("Froth Motion Vectors", visual)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # %% Original cell 7 (markdown)
    # Stability parameters

    # %% Original cell 8 (code)


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

    # %% Original cell 9 (code)
    # After running block matching
    speed, vx, vy, motion_vectors = calculate_froth_speed_block_matching(img1, img2)

    # Extract dx, dy from motion vectors
    vx_list = [mv[2] for mv in motion_vectors]
    vy_list = [mv[3] for mv in motion_vectors]

    stability = calculate_stability_from_block_matching(vx_list, vy_list)
    print("Stability Scores:", stability)

    # %% Original cell 10 (markdown)
    # test for 20 frames  

    # %% Original cell 11 (code)
    import cv2
    frames = []
    for i in range(20):
        path = f"image{i+1}.tiff"  # Adjust filename pattern if needed
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Warning: Could not load {path}")
            continue
        frames.append(img)

    # %% Original cell 12 (code)
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

    # %% Original cell 13 (code)
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

    # %% Original cell 14 (markdown)
    # plot Froth Speed Over Time and  Directional Stability Over Time

    # %% Original cell 15 (code)
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

    # %% Original cell 16 (code)
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

    # %% Original cell 17 (markdown)
    # ### test

    # %% Original cell 18 (code)

    # %% Original cell 19 (code)
    import cv2
    import numpy as np

    # Start video capture from default camera
    cap = cv2.VideoCapture(0)

    # Read the first frame and convert to grayscale
    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255  # full saturation for vivid colors

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert flow to polar coordinates (magnitude and angle)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
        # Use hue for direction, value for magnitude
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV to BGR for display
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Show the optical flow visualization
        cv2.imshow('Optical Flow', rgb_flow)
        cv2.imshow('Camera Feed', frame)

        # Update previous frame
        prev_gray = gray

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # %% Original cell 20 (code)
    import numpy as np
    import cv2 as cv

    # Capture from default camera
    cap = cv.VideoCapture(0)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=1,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    if not ret:
        print("Failed to grab initial frame from camera!")
        exit()

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame grabbed from camera!")
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

            img = cv.add(frame, mask)
            cv.imshow('Lucas-Kanade Optical Flow', img)

            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        k = cv.waitKey(30) & 0xff
        if k == 27:  # ESC to exit
            break

    cap.release()
    cv.destroyAllWindows()

    # %% Original cell 21 (code)
    import cv2
    import numpy as np

    # --- Configuration ---
    IMAGE_PATH = 'F61 (19).tiff' # Put the path to your image here
    points = []

    # --- Mouse callback function ---
    def get_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                print(f"Point {len(points)} selected: ({x}, {y})")

    # --- Main script ---
    img = cv2.imread(IMAGE_PATH)
    clone = img.copy()
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", get_points)

    print("Instructions: Click two points on the image to measure the distance.")

    while True:
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF

        # If two points have been selected, draw the line and calculate distance
        if len(points) == 2:
            cv2.line(img, points[0], points[1], (0, 255, 0), 2)
        
            # Calculate Euclidean distance
            dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
        
            # Put the text on the screen
            text = f"Distance: {dist:.2f} pixels"
            cv2.putText(img, text, (points[0][0] - 20, points[0][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(text)
            points = [] # Reset for next measurement

        # Press 'r' to reset the image
        if key == ord('r'):
            img = clone.copy()
            points = []

        # Press 'q' to quit
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # %% Original cell 22 (code)
    import cv2
    import numpy as np

    # Load your two grayscale images (image1 and image2)
    # Make sure they are the same size

    def calculate_froth_speed_optical_flow(image1, image2):
        """Calculates dense optical flow using Farneback's algorithm."""
    
        # Calculate the optical flow
        # This returns a 2-channel array with the (dx, dy) displacement for each pixel
        flow = cv2.calcOpticalFlowFarneback(
            prev=image1, 
            next=image2, 
            flow=None, 
            pyr_scale=0.5, 
            levels=3, 
            winsize=15, 
            iterations=3, 
            poly_n=5, 
            poly_sigma=1.2, 
            flags=0
        )

        # Calculate the magnitude (speed) and angle of the flow for visualization
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
        # --- Create a visualization of the flow ---
        hsv = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Angle gives color
        hsv[..., 1] = 255  # Full saturation
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) # Speed gives brightness
    
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
        # --- Calculate average speed ---
        # Here, 'magnitude' holds the pixel displacement for every pixel
        # You can apply your spatial_scale and temporal_scale to this
        avg_speed_pixels = np.mean(magnitude)

        return avg_speed_pixels, bgr_flow

    # --- Example Usage ---
    # Assuming image1 and image2 are loaded
    avg_speed_px, flow_visualization = calculate_froth_speed_optical_flow(img1, img2)

    cv2.imshow("Original Image", img1)
    cv2.imshow("Optical Flow Visualization", flow_visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Average motion in pixels per frame: {avg_speed_px:.2f}")

    # %% Original cell 23 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
