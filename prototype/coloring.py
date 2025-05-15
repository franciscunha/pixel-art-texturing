
import cv2
import numpy as np


def get_similar_color(base: cv2.Mat, rect: tuple[int, int, int, int], hsv_shift: tuple[int, int, int]):
    """
    Gets the average color of a region of an image, then shifts it.

    Args:
        base: Image to extract color from
        rect: Boundaries of image region, layout is x y w h
        hsv_shift: How much to shift the color by, in HSV color space

    Returns: The average color, shifted, in BGR color space
    """
    y, x, h, w = rect
    # Get the region of interest and its average color
    roi_per_channel = [base[y: y + h, x: x + w, i] for i in range(4)]
    mean_color_bgr = [int(np.round(np.mean(roi_per_channel[i])))
                      for i in range(4)]

    # Convert BGR to HSV
    mean_color_hsv = cv2.cvtColor(
        np.uint8([[mean_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # Apply the shift converting to int32 to allow for negative values in parameter
    shifted_hsv = \
        mean_color_hsv.astype(np.int32) + np.array(hsv_shift, dtype=np.int32)

    # Wrap around H channel
    shifted_hsv[0] = shifted_hsv[0] % 180

    # Clip S and V channels to valid range
    shifted_hsv[1] = np.clip(shifted_hsv[1], 0, 255)
    shifted_hsv[2] = np.clip(shifted_hsv[2], 0, 255)

    # Convert back to uint8 for OpenCV
    shifted_hsv_uint8 = shifted_hsv.astype(np.uint8)

    # Conversions are for images, so we use single pixel images for the conversions
    # Extract that pixel's color to return
    shifted_color_bgr = cv2.cvtColor(
        np.uint8([[shifted_hsv_uint8]]), cv2.COLOR_HSV2BGR)[0][0]

    return shifted_color_bgr


def monochromize_image(img: cv2.Mat, color: np.ndarray):
    if color.shape != (3,):
        raise ValueError(f"{color} is not a color")
    img[:, :, :3] = color
