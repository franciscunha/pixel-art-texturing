from math import ceil
import cv2
import numpy as np


def draw_arrow(
    img: cv2.Mat,
    start: np.array,
    end: np.array,
    color: np.array,
    size=5,
    draw_tip=True,
    big_tip=False
):
    """
    Draw an arrow from start to end on the image.
    start and end should be in (x,y) format for OpenCV compatibility.
    """
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("Wrong input shape")

    # OpenCV expects points as (x,y) tuples
    start_point = (int(start[0]), int(start[1]))
    end_point = (int(end[0]), int(end[1]))

    cv2.line(img, start_point, end_point, color, size)

    if not draw_tip:
        return

    alpha = 5 if big_tip else 3
    beta = 4 if big_tip else 2

    # Get points that form triangle's tip
    front_dir = ((end - start) / np.linalg.norm(end - start))
    side_dir = np.array([front_dir[1], -front_dir[0]])  # 90 deg rot front_dir
    tip = np.array([[end + (alpha*size) * front_dir],
                    [end + (beta*size) * side_dir],
                    [end - (beta*size) * side_dir]], dtype=np.int32)

    cv2.fillPoly(img, [tip], color)


def draw_vector_field(
    vec_field: np.array,
    input_vector_coords: list[tuple[int, int]] = [],
    scalar_field: np.array = None,
    cell_size: int = 24
):
    """
    Visualize a vector field. The input array should have shape (h, w, 2),
    where vec_field[y, x] contains the (dx, dy) vector at position (x, y).
    """
    h, w, vec_shape = vec_field.shape
    if vec_shape != 2:
        raise ValueError("Expected shape (h, w, 2)")

    arrow_size = ceil(cell_size / 12)
    center_offset = cell_size / 2
    img = np.full((h * cell_size, w * cell_size, 4), 255, np.uint8)
    # img = draw_grid(cell_size, h, w)

    for y in range(h):
        for x in range(w):
            vec = vec_field[y, x]  # This is (dx, dy)

            if np.all(vec == 0):
                continue

            # Set vector length to 1.1
            vec = vec * (1.1 / np.linalg.norm(vec))

            # Draw only some of the arrow tips
            draw_tip = x % 3 == 0 and y % 3 == 0

            # Convert grid position to pixel coordinates (x,y format for OpenCV)
            start = np.array([
                (x * cell_size) + center_offset,  # x coordinate
                (y * cell_size) + center_offset   # y coordinate
            ], dtype=np.float64)

            # The vector is in (dx, dy) format, which matches OpenCV's (x,y) expectation
            # Scale the vector and add to start to get the end point
            end = np.array([
                start[0] + vec[0] * center_offset,
                start[1] + vec[1] * center_offset
            ], dtype=np.float64)

            # Move start to the pixel's corner
            start += start - end

            color = (0, 0, 255) \
                if (y, x) in input_vector_coords or input_vector_coords == [] \
                else (255, 0, 0)

            draw_arrow(img, start, end, color, arrow_size, draw_tip, True)

            if scalar_field is not None:
                start_point = (
                    int(start[0]) - int(center_offset/2),
                    int(start[1]) - int(center_offset/2)
                )
                cv2.putText(
                    img, f"{scalar_field[y, x]:.2f}",
                    start_point, 1, 1, (0, 0, 255))

    return img
