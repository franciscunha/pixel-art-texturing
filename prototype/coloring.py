
import cv2
import numpy as np


def change_color_space(color, cv2_conversion_code):
    if len(color) != 3 and len(color) != 4:
        raise ValueError("Color should be an array-like with 3 or 4 elements")
    # Conversions are for images, so we use single pixel images for the conversions
    # Extract that pixel's color to return
    return cv2.cvtColor(np.uint8([[color]]), cv2_conversion_code)[0][0]


def area_mode_color(base: cv2.Mat, rect: tuple[int, int, int, int]):
    y, x, h, w = rect
    roi = base[y: y + h, x: x + w, :]
    colors, counts = np.unique(
        roi.reshape(-1, roi.shape[-1]), axis=0, return_counts=True)
    mode_index = np.argmax(counts)
    return colors[mode_index]


def area_mean_color(base: cv2.Mat, rect: tuple[int, int, int, int]):
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
    mean_color_bgr = area_mean_color(base, rect)
    mean_color_hsv = change_color_space(mean_color_bgr, cv2.COLOR_BGR2HSV)

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

    shifted_color_bgr = change_color_space(
        shifted_hsv_uint8, cv2.COLOR_HSV2RGB)

    return shifted_color_bgr


def monochromize_image(img: cv2.Mat, color: np.ndarray):
    if color.shape != (3,) and color.shape != (4,):
        raise ValueError(f"{color} is not a color")
    img[:, :, :3] = color[:3]


def extract_palette(img: cv2.Mat):
    palette = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
    no_transparent = palette[palette[:, 3] > 0]
    return no_transparent


def find_closest_color(target: np.ndarray, palette: np.ndarray, exclude: np.ndarray = []):
    """
    Find color in palette with lowest Euclidean distance to target color in CIELab space.
    """
    # Euclidean distance is acceptable if we use a uniform color space
    # https://en.wikipedia.org/wiki/Color_difference#Uniform_color_spaces

    best_distance = np.inf
    closest_color = None

    target_uniform = change_color_space(target, cv2.COLOR_BGR2LUV)
    palette_uniform = [change_color_space(color, cv2.COLOR_BGR2LUV)
                       for color in palette]

    for i in range(len(palette_uniform)):
        if np.any(np.all(palette[i] == exclude, axis=1)):
            # equivalent to `palette[i] in exlude`
            # if using python lists instead of ndarrays
            continue
        distance = np.linalg.norm(target_uniform - palette_uniform[i])
        if distance >= best_distance:
            continue
        best_distance = distance
        closest_color = palette_uniform[i]

    return change_color_space(closest_color, cv2.COLOR_LUV2BGR)


if __name__ == "__main__":
    base_file = "data/shaded_tree.png"
    base = cv2.imread(base_file, cv2.IMREAD_UNCHANGED)

    palette = extract_palette(base)
