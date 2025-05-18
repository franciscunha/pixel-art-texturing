
import cv2
import numpy as np


def area_average_color(base: cv2.Mat, rect: tuple[int, int, int, int]):
    y, x, h, w = rect
    # Get the region of interest and its average color
    roi_per_channel = [base[y: y + h, x: x + w, i] for i in range(4)]
    return [int(np.round(np.mean(roi_per_channel[i]))) for i in range(4)]


def get_shifted_color(base: cv2.Mat, rect: tuple[int, int, int, int], hsv_shift: tuple[int, int, int]):
    """
    Gets the average color of a region of an image, then shifts it.

    Args:
        base: Image to extract color from
        rect: Boundaries of image region, layout is x y w h
        hsv_shift: How much to shift the color by, in HSV color space

    Returns: The average color, shifted, in BGR color space
    """
    mean_color_bgr = area_average_color(base, rect)

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
    if color.shape != (3,) and color.shape != (4,):
        raise ValueError(f"{color} is not a color")
    img[:, :, :3] = color[:3]


def extract_palette(img: cv2.Mat):
    return np.unique(img.reshape(-1, img.shape[-1]), axis=0)


def find_closest_color(target: np.ndarray, palette: np.ndarray, exclude: np.ndarray = []):
    """
    Find color in palette with lowest Euclidean distance to target color.
    """
    # Euclidean distance seems to be a metric that is generally used
    # (see https://en.wikipedia.org/wiki/Color_difference#sRGB) but I
    # wonder if there's something better (which I can cite!)

    # TODO turns out it doesn't work as well as I'd expect
    # TODO I could maybe get the palette only from within the boundary,
    # TODO or convert to HSV and find closest prioritizing V then S then H

    best_distance = np.inf
    closest_color = None

    for color in palette:
        if np.any(np.all(color == exclude, axis=1)):
            # equivalent to `color in exlude` if using python lists instead of ndarrays
            continue
        distance = np.linalg.norm(target - color)
        if distance >= best_distance:
            continue
        best_distance = distance
        closest_color = color

    return closest_color


if __name__ == "__main__":
    base_file = "data/shaded_tree.png"
    base = cv2.imread(base_file, cv2.IMREAD_UNCHANGED)

    palette = extract_palette(base)
