import cv2
import numpy as np


# Shared helpers

def flood_fill_mask(image: cv2.Mat, start_point: tuple[int, int]):
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    _, _, mask_padded, _ = cv2.floodFill(
        bgr, None, start_point, (255, 255, 255))
    mask = mask_padded[1:-1, 1:-1]
    return mask


# Color difference

def bgr_to_lab(color):
    if len(color) != 3 and len(color) != 4:
        raise ValueError("Color should be an array-like with 3 or 4 elements")

    # Make input be a float32 pixel, otherwise we lose precision
    # on lab and need to convert it later
    # 8-bit images: L←L∗255/100,a←a+128,b←b+128
    # 32-bit images: L, a, and b are left as is
    # see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    bgr_pixel = np.array(
        [[[color[0]/255.0, color[1]/255.0, color[2]/255.0]]], dtype=np.float32)

    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)

    # Conversions are for images, so we use single pixel images for the
    # conversions. Extract that pixel's color to return.
    return lab_pixel[0][0]


def find_closest_color(
    target: np.ndarray,
    palette: np.ndarray
):
    """
    Find color in palette with lowest Euclidean distance to 
    target color in CIELab space.
    """
    # Euclidean distance is acceptable if we use a uniform color space
    # https://en.wikipedia.org/wiki/Color_difference#Uniform_color_spaces

    best_distance = np.inf
    closest_color_index = -1

    target_uniform = bgr_to_lab(target[:3])
    palette_uniform = [bgr_to_lab(color[:3]) for color in palette]

    for i in range(len(palette_uniform)):
        distance = np.linalg.norm(target_uniform - palette_uniform[i])
        # distance close to zero -> same color, which we skip
        if distance >= best_distance or distance <= 1e-4:
            continue
        best_distance = distance
        closest_color_index = i

    return palette[closest_color_index]


def extract_palette(img: cv2.Mat, exclude: np.ndarray = np.empty((0, 4), np.uint8)):
    palette = np.unique(img.reshape(-1, img.shape[-1]), axis=0)

    exclude_mask = ~(palette[:, None] == exclude).all(-1).any(1)
    not_excluded = palette[exclude_mask]

    no_transparent = not_excluded[not_excluded[:, 3] > 0]
    return no_transparent


def color_map_by_similarity(
    image: cv2.Mat,
    mask: np.ndarray,
    exclude: np.ndarray = np.empty((0, 4), np.uint8),
):
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask don't match")

    filled = np.zeros_like(mask, dtype=np.uint8)
    colors = np.copy(image)
    palette = extract_palette(image, exclude)
    ys, xs = np.where(mask)

    for i in range(len(ys)):
        y, x = ys[i], xs[i]

        if filled[y, x] > 0:
            continue

        target = image[y, x]
        color = find_closest_color(target, palette)

        flood_mask = flood_fill_mask(image, (x, y))

        colors[flood_mask == True, :] = color
        filled = filled + flood_mask

    return colors


# Shared border

def color_frequency(image: cv2.Mat):
    return np.unique(
        image.reshape(-1, image.shape[-1]),
        axis=0,
        return_counts=True
    )


def mode_color(image: cv2.Mat, exclude: np.ndarray = np.empty((0, 4), np.uint8)):
    colors, counts = color_frequency(image)
    # Color indices sorted by count, decreasing
    sorted_indices = np.flip(np.argsort(counts))

    # Iterate through indices so we can skip transparent colors
    for i in sorted_indices:
        if colors[i][3] == 0:
            # is transparent
            continue
        if len(exclude) > 0 and np.any(np.all(colors[i] == exclude, axis=1)):
            # is in exclude list
            continue
        return colors[i]

    # If all colors are transparent, return the most frequent anyway
    return colors[sorted_indices[0]]


def dominant_border_color(
    image: cv2.Mat,
    start_point: tuple[int, int],
    exclude: np.ndarray = np.empty((0, 4), np.uint8),
):
    mask_flooded_region = flood_fill_mask(image, start_point)
    mask_flooded_and_borders = cv2.dilate(mask_flooded_region, np.ones((3, 3)))
    mask_borders_only = cv2.bitwise_xor(
        mask_flooded_region, mask_flooded_and_borders)

    # Mask out the original image
    borders = cv2.bitwise_and(image, image, mask=mask_borders_only)

    # Most frequent color in borders, mask for pixels where this applies
    return mode_color(borders, exclude), mask_flooded_region


def color_map_by_shared_border(
    image: cv2.Mat,
    mask: np.ndarray,
    exclude: np.ndarray = np.empty((0, 4), np.uint8),
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


# HSV shift

def change_color_space(color, cv2_conversion_code):
    if len(color) != 3 and len(color) != 4:
        raise ValueError("Color should be an array-like with 3 or 4 elements")
    # Conversions are for images, so we use single pixel images for the
    # conversions. Extract that pixel's color to return.
    return cv2.cvtColor(np.uint8([[color]]), cv2_conversion_code)[0][0]


def get_shifted_color(
        bgr: tuple[int, int, int],
        hsv_shift: tuple[int, int, int]
):
    hsv = change_color_space(bgr, cv2.COLOR_BGR2HSV)

    # Apply the shift converting to int32 to allow
    # for negative values in parameter
    shifted_hsv = \
        hsv.astype(np.int32) + np.array(hsv_shift, dtype=np.int32)

    # Wrap around H channel
    shifted_hsv[0] = shifted_hsv[0] % 180

    # Clip S and V channels to valid range
    shifted_hsv[1] = np.clip(shifted_hsv[1], 0, 255)
    shifted_hsv[2] = np.clip(shifted_hsv[2], 0, 255)

    # Convert back to uint8 for OpenCV, then into BGR
    shifted_color_bgr = change_color_space(
        shifted_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return shifted_color_bgr


def color_map_by_hsv_shift(
        image: cv2.Mat,
        mask: np.ndarray,
        hsv_shift: tuple[int, int, int]
):
    colors = np.copy(image)
    ys, xs = np.where(mask)

    for i in range(len(ys)):
        y, x = ys[i], xs[i]

        colors[y, x, :3] = get_shifted_color(colors[y, x, :3], hsv_shift)

    return colors


# Auto color map

def distances_in_color_map(
    image: cv2.Mat,
    color_map: cv2.Mat,
    mask: np.ndarray,
):
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask don't match")

    filled = np.zeros_like(mask, dtype=np.uint8)
    ys, xs = np.where(mask)

    distances = []

    for i in range(len(ys)):
        y, x = ys[i], xs[i]

        if filled[y, x] > 0:
            continue

        original = image[y, x]
        mapped_to = color_map[y, x]

        distance = np.linalg.norm(
            bgr_to_lab(original[:3]) - bgr_to_lab(mapped_to[:3]))

        distances.append(distance)

        flood_mask = flood_fill_mask(image, (x, y))
        filled = filled + flood_mask

    return distances


def color_map_auto(
    image: cv2.Mat,
    mask: np.ndarray,
    hsv_shift: tuple[int, int, int],
    max_distance: float,
    exclude: np.ndarray = np.empty((0, 4), np.uint8),
):
    # If the image has a single color, use HSV shift
    palette = extract_palette(image, exclude)
    if len(palette) == 1:
        print("Using HSV")
        return color_map_by_hsv_shift(image, mask, hsv_shift)

    # Try shared border
    map = color_map_by_shared_border(image, mask, exclude)
    # Validate it
    distances = distances_in_color_map(image, map, mask)
    if max(distances) <= max_distance:
        print("Using shared border")
        return map

    # If jumps are too large, use color difference
    print(f"Using color difference")
    return color_map_by_similarity(image, mask, exclude)


# Main export

def color_map(
    image: cv2.Mat,
    mask: np.ndarray,
    exclude: np.ndarray = np.empty((0, 4), np.uint8),
    type: str = "auto",
    max_distance: float | None = 10,
    hsv_shift: tuple[int, int, int] | None = None
):
    if type == "auto":
        map = color_map_auto(image, mask, hsv_shift, max_distance, exclude)
    elif type == "similarity":
        map = color_map_by_similarity(image, mask, exclude)
    elif type == "border":
        map = color_map_by_shared_border(image, mask, exclude)
    elif type == "hsv":
        map = color_map_by_hsv_shift(image, mask, hsv_shift)
    else:
        raise ValueError(
            "Type must be 'auto', 'similarity', 'border' or 'hsv'")
    return cv2.bitwise_and(map, map, mask=mask.astype(np.uint8))
