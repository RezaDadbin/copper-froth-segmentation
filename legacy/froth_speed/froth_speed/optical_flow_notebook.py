"""Historical source export; maintained entry points live in scripts/.

Original: codes/froth_speed/froth_speed/optical flow.ipynb
Importing this file does not train, install packages, rename files or render.
historical_experiment() is a manual research transcript, not a supported pipeline.
"""


def historical_experiment():
    # %% Original cell 0 (code)
    import cv2
    import numpy as np
    from scipy.ndimage import median_filter

    def calculate_optical_flow_speed(image1, image2, spatial_scale=1.0, temporal_scale=1.0):
        """
        Calculates average speed and directional components between two grayscale images.
    
        Parameters:
            image1 (np.ndarray): First grayscale image.
            image2 (np.ndarray): Second grayscale image.
            spatial_scale (float): Scale factor for spatial resolution (e.g., mm/pixel).
            temporal_scale (float): Time between frames (e.g., seconds).
    
        Returns:
            speed (float): Average speed magnitude.
            speed_x (float): Horizontal speed component.
            speed_y (float): Vertical speed component.
        """
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(image1, image2, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Filter out noise
        flow[..., 0] = median_filter(flow[..., 0], size=5)
        flow[..., 1] = median_filter(flow[..., 1], size=5)

        # Compute magnitude and direction
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Normalize direction vectors
        norm_flow_x = flow[..., 0] / (magnitude + 1e-5)
        norm_flow_y = flow[..., 1] / (magnitude + 1e-5)

        # Compute average speed
        avg_magnitude = np.mean(magnitude)
        speed_x = np.mean(norm_flow_x) * avg_magnitude
        speed_y = np.mean(norm_flow_y) * avg_magnitude

        # Apply scaling
        speed = avg_magnitude * spatial_scale / temporal_scale
        speed_x *= spatial_scale / temporal_scale
        speed_y *= spatial_scale / temporal_scale

        return speed, speed_x, speed_y

    # %% Original cell 1 (code)
    import cv2
    frame1 = cv2.imread('image1.tiff', cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread('image2.tiff', cv2.IMREAD_GRAYSCALE)

    # %% Original cell 2 (code)
    speed, vx, vy = calculate_optical_flow_speed(frame1, frame2, spatial_scale=0.1, temporal_scale=0.04)
    print(f"Speed: {speed:.2f} units/s, X: {vx:.2f}, Y: {vy:.2f}")

    # %% Original cell 3 (code)

    pass


if __name__ == "__main__":
    raise SystemExit("Historical transcript only. Use the linked scripts/ commands in README.md.")
