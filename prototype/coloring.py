
import cv2
import numpy as np

from boundaries import mask_from_boundary
from helpers import flood_fill_mask
from visualizations import show_scaled


def change_color_space(color, cv2_conversion_code):
    if len(color) != 3 and len(color) != 4:
        raise ValueError("Color should be an array-like with 3 or 4 elements")
    # Conversions are for images, so we use single pixel images for the
    # conversions. Extract that pixel's color to return.
    return cv2.cvtColor(np.uint8([[color]]), cv2_conversion_code)[0][0]


def color_frequency(image: cv2.Mat):
    return np.unique(
        image.reshape(-1, image.shape[-1]),
        axis=0,
        return_counts=True
    )


def mode_color(image: cv2.Mat, exclude: np.ndarray = []):
    colors, counts = color_frequency(image)
    # Color indices sorted by count, decreasing
    sorted_indices = np.flip(np.argsort(counts))

    # Iterate through indices so we can skip transparent colors
    for i in sorted_indices:
        # if not transparent and not in exclude list
        if colors[i][3] > 0 and not np.any(np.all(colors[i] == exclude, axis=1)):
            return colors[i]

    # If all colors are transparent, return the most frequent anyway
    return colors[sorted_indices[0]]


def area_mode_color(base: cv2.Mat, rect: tuple[int, int, int, int]):
    y, x, h, w = rect
    roi = base[y: y + h, x: x + w, :]
    return mode_color(roi)


def area_mean_color(base: cv2.Mat, rect: tuple[int, int, int, int]):
    y, x, h, w = rect
    # Get the region of interest and its average color
    roi_per_channel = [base[y: y + h, x: x + w, i] for i in range(4)]
    return [int(np.round(np.mean(roi_per_channel[i]))) for i in range(4)]


def get_shifted_color(
        base: cv2.Mat,
        rect: tuple[int, int, int, int],
        hsv_shift: tuple[int, int, int]
):
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

    # Apply the shift converting to int32 to allow
    # for negative values in parameter
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
    if len(color) != 3 and len(color) != 4:
        raise ValueError(f"{color} is not a color")
    img[:, :, :3] = color[:3]


def extract_palette(img: cv2.Mat):
    palette = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
    no_transparent = palette[palette[:, 3] > 0]
    return no_transparent


def find_closest_color(
    target: np.ndarray,
    palette: np.ndarray,
    exclude: np.ndarray = []
):
    """
    Find color in palette with lowest Euclidean distance to 
    target color in CIELab space.
    """
    # Euclidean distance is acceptable if we use a uniform color space
    # https://en.wikipedia.org/wiki/Color_difference#Uniform_color_spaces

    best_distance = np.inf
    closest_color = None

    target_uniform = change_color_space(target, cv2.COLOR_BGR2LAB)
    palette_uniform = [change_color_space(color, cv2.COLOR_BGR2LAB)
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

    color_with_alpha = np.zeros((4), dtype=np.uint8)
    color_with_alpha[:3] = change_color_space(closest_color, cv2.COLOR_LAB2BGR)
    color_with_alpha[3] = 255
    return color_with_alpha


def dominant_border_color(
    image: cv2.Mat,
    start_point: tuple[int, int],
    exclude: np.ndarray = []
):
    mask_flooded_region = flood_fill_mask(image, start_point)
    mask_flooded_and_borders = cv2.dilate(mask_flooded_region, np.ones((3, 3)))
    mask_borders_only = cv2.bitwise_xor(
        mask_flooded_region, mask_flooded_and_borders)

    # Mask out the original image
    borders = cv2.bitwise_and(image, image, mask=mask_borders_only)

    # Most frequent color in borders, mask for pixels where this applies
    return mode_color(borders, exclude), mask_flooded_region


def color_map_by_similarity(
    image: cv2.Mat,
    mask: np.ndarray,
    exclude: np.ndarray = []
):
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask don't match")

    filled = np.zeros_like(mask, dtype=np.uint8)
    colors = np.copy(image)
    palette = extract_palette(image)
    ys, xs = np.where(mask)

    for i in range(len(ys)):
        y, x = ys[i], xs[i]

        if filled[y, x] > 0:
            continue

        target = image[y, x]
        color = find_closest_color(
            target, palette, exclude=np.append(exclude, [target], axis=0))

        flood_mask = flood_fill_mask(image, (x, y))

        colors[flood_mask == True, :] = color
        filled = filled + flood_mask

    return colors


def color_map_by_shared_border(
    image: cv2.Mat,
    mask: np.ndarray,
    exclude: np.ndarray = []
):
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask don't match")

    filled = np.zeros_like(mask, dtype=np.uint8)
    colors = np.copy(image)
    ys, xs = np.where(mask)

    for i in range(len(ys)):
        y, x = ys[i], xs[i]

        if filled[y, x] > 0:
            continue

        color, flood_mask = dominant_border_color(image, (x, y), exclude)

        colors[flood_mask == True, :] = color
        filled = filled + flood_mask

    return colors


def color_map(
    image: cv2.Mat,
    mask: np.ndarray,
    exclude: np.ndarray = [],
    type: str = "border"
):
    if type == "similarity":
        return color_map_by_similarity(image, mask, exclude)
    elif type == "border":
        return color_map_by_shared_border(image, mask, exclude)
    else:
        raise ValueError("Type must be 'similarity' or 'border'")


if __name__ == "__main__":
    base = cv2.imread("data/bases/green_sphere.png", cv2.IMREAD_UNCHANGED)
    mask = mask_from_boundary(
        cv2.imread("data/bases/green_sphere.png", cv2.IMREAD_UNCHANGED))

    scale = 12

    show_scaled("Original", base, scale)

    mapped_border = color_map(base, mask, type="border")
    mapped_similarity = color_map(base, mask, type="similarity")

    show_scaled("Border", mapped_border, scale)
    show_scaled("Similarity", mapped_similarity, scale)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
