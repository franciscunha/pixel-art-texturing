import cv2
import numpy as np

from src.color import mode_color
from src.orientation.vectors import area_vector, quantize_direction


def find_element(
    vector_field: np.ndarray,
    point: tuple[int, int],
    elements: np.ndarray
):
    x, y = point
    rows, cols, h, w, _ = elements.shape

    if rows != 3 or cols != 3:
        raise ValueError("Elements should be within a 3x3 matrix")

    # Find the direction of that area
    area_dir = area_vector(vector_field, (y, x, w, h))

    # Choose appropriate element according to direction
    element_dir_x, element_dir_y = quantize_direction(area_dir)

    # Deal with placements without a direction
    if element_dir_y == 0 and element_dir_x == 0:
        raise ValueError(f"Point {point} has no direction in vector field")

    return elements[element_dir_x+1, element_dir_y+1]


def place_element(
        destination: cv2.Mat,
        element: cv2.Mat,
        point: tuple[int, int],
        mask: np.ndarray,
        color_map: cv2.Mat | None = None,
        type: str = "region"
):

    x0, y0 = point

    # Boundary check
    h, w = element.shape[:2]
    if y0 + h > destination.shape[0] or x0 + w > destination.shape[1]:
        return False

    # Find element color

    if type == "region":
        region_of_interest = color_map[y0: y0 + h, x0: x0 + w, :]
        region_color = mode_color(region_of_interest)

    # Placing

    for x in range(x0, x0 + w):
        for y in range(y0, y0 + h):
            if not mask[y, x]:
                continue

            alpha = element[y-y0, x-x0, 3]

            # TODO proper alpha blending
            if alpha <= 10:
                continue

            if type == "region":
                destination[y, x, :] = region_color
            elif type == "per-pixel":
                destination[y, x, :] = color_map[y, x, :]
            else:
                raise ValueError("Type must be 'per-pixel' or 'region'")

    return True


def place_elements(
    # Input
    source, elements, mask,
    # Pre processed data
    positions, color_map, vector_field,
    # Parameters
    element_color_mode="region"
):
    result = source.copy()
    for position in positions:
        element = find_element(vector_field, position, elements)
        place_element(result, element, position, mask,
                      color_map, element_color_mode)
    return result
