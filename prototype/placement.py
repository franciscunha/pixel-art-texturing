import cv2

from coloring import dominant_border_color, get_shifted_color, monochromize_image


def place_pattern(
        destination: cv2.Mat,
        pattern: cv2.Mat,
        y: int, x: int,
        hsv_shift: tuple[int, int, int] | None = None,
        color_map: cv2.Mat | None = None):

    # Boundary check
    h, w = pattern.shape[:2]
    if y + h > destination.shape[0] or x + w > destination.shape[1]:
        return False

    region_of_interest = destination[y: y + h, x: x + w, :]

    # Find pattern color
    if hsv_shift is None and color_map is None:
        raise ValueError("One of hsv_shift or color_map needs to be set")

    if hsv_shift is not None:
        color = get_shifted_color(destination, (y, x, h, w), hsv_shift)
    if color_map is not None:
        color = color_map[y, x]
    monochromize_image(pattern, color)

    # TODO pattern should inherit base's alpha in final placement
    # Extract alpha
    pattern_alpha = pattern[:, :, 3] / 255.0

    # Placing
    for c in range(3):
        # implicit double for loop due to numpy
        destination[y: y + h, x: x + w, c] = \
            (1 - pattern_alpha) * region_of_interest[:, :, c] \
            + pattern_alpha * pattern[:, :, c]

    return True
