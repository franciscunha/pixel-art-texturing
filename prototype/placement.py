import cv2

from coloring import area_mode_color, dominant_border_color, extract_palette, find_closest_color, get_shifted_color, monochromize_image


def place_pattern(destination: cv2.Mat, pattern: cv2.Mat, y: int, x: int, hsv_shift: tuple[int, int, int] | None = None):

    # Boundary check
    h, w = pattern.shape[:2]
    if y + h > destination.shape[0] or x + w > destination.shape[1]:
        return False

    region_of_interest = destination[y: y + h, x: x + w, :]

    # Extract alpha
    pattern_alpha = pattern[:, :, 3] / 255.0
    if hsv_shift is None:
        color = dominant_border_color(destination, (x, y))

        # TODO fix: placing a pattern affects the border colors

        # ! Uncomment for color difference
        target = area_mode_color(destination, (y, x, h, w))
        # ! TODO restructure code so that this call isn't here, because it's an easy optimization to only do this once instead of 500 times
        # palette = extract_palette(destination)
        # color = find_closest_color(target, palette, exclude=[target])

        print(
            f"Placed pattern with color {color} at x,y: {(x, y)} whose color is {destination[y, x, :]} and mode is {target}")
    else:
        color = get_shifted_color(destination, (y, x, h, w), hsv_shift)
    monochromize_image(pattern, color)

    # TODO pattern should inherit base's alpha in final placement

    # Placing
    for c in range(3):
        # implicit double for loop due to numpy
        destination[y: y + h, x: x + w, c] = \
            (1 - pattern_alpha) * region_of_interest[:, :, c] \
            + pattern_alpha * pattern[:, :, c]

    return True
