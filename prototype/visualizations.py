import cv2
import numpy as np


# Written by Claude
def draw_grid(cell_size, grid_height, grid_width, background_color=(255, 255, 255), line_color=(0, 0, 0), line_thickness=1):
    # Calculate the pixel dimensions
    img_height = grid_height * cell_size
    img_width = grid_width * cell_size

    # Create a blank image with the specified background color
    grid_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * \
        np.array(background_color, dtype=np.uint8)

    # Draw vertical lines
    for i in range(grid_width + 1):
        x = i * cell_size
        cv2.line(grid_image, (x, 0), (x, img_height),
                 line_color, line_thickness)

    # Draw horizontal lines
    for i in range(grid_height + 1):
        y = i * cell_size
        cv2.line(grid_image, (0, y), (img_width, y),
                 line_color, line_thickness)

    return grid_image


def visualize_vector_field(vec_field: np.array, scalar_field: np.array = None, cell_size: int = 24):
    """
    Visualize a vector field. The input array should have shape (h, w, 2),
    where vec_field[y, x] contains the (dx, dy) vector at position (x, y).
    """
    h, w, vec_shape = vec_field.shape
    if vec_shape != 2:
        raise ValueError("Expected shape (h, w, 2)")

    center_offset = cell_size / 2
    # img = np.zeros((h * cell_size, w * cell_size, 4), np.uint8)
    img = draw_grid(cell_size, h, w)

    for y in range(h):
        for x in range(w):
            vec = vec_field[y, x]  # This is (dx, dy)

            if np.all(vec == 0):
                continue

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

            draw_arrow(img, start, end, (255, 0, 0), 3)

            if scalar_field is not None:
                start_point = (
                    int(start[0]) - int(center_offset/2), int(start[1]) - int(center_offset/2))
                cv2.putText(
                    img, f"{scalar_field[y, x]:.2f}", start_point, 1, 1, (0, 0, 255))

    return img


def draw_arrow(img: cv2.Mat, start: np.array, end: np.array, color: np.array, size=5):
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

    # Get points that form triangle's tip
    front_dir = ((end - start) / np.linalg.norm(end - start))
    side_dir = np.array([front_dir[1], -front_dir[0]])  # 90 deg rot front_dir
    tip = np.array([[end + (3*size) * front_dir], [end + (2*size) * side_dir],
                   [end - (2*size) * side_dir]], dtype=np.int32)

    cv2.fillPoly(img, [tip], color)


def visualize_mask(mask, original_shape):
    """Helper function to visualize the availability mask for debugging"""
    vis_mask = np.zeros_like(original_shape)
    vis_mask[:, :, 3] = mask.astype(np.uint8) * 255  # Alpha channel
    vis_mask[:, :, 0] = mask.astype(np.uint8) * 255  # Blue
    vis_mask[:, :, 1] = 0  # Green
    vis_mask[:, :, 2] = 0  # Red
    return vis_mask


def show_scaled(title: str, img: cv2.Mat, factor: int):
    scaled = cv2.resize(img, dsize=None, fx=factor,
                        fy=factor, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(title, scaled)
