from math import ceil
import cv2
import numpy as np


def show_scaled(title: str, img: cv2.Mat, factor: int):
    scaled = cv2.resize(img, dsize=None, fx=factor,
                        fy=factor, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(title, scaled)


def save_scaled(filepath: str, img: cv2.Mat, factor: int):
    scaled = cv2.resize(img, dsize=None, fx=factor,
                        fy=factor, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filepath, scaled)


def highlight_pixels(
    image: cv2.Mat,
    scale_factor: int,
    pixel_coords: list[tuple[int, int]],
    border_thickness=1, border_color=(0, 0, 255)
):
    """
    Scale an image and place borders around specified pixels.
    """
    # Validate inputs
    if image is None or image.size == 0:
        raise ValueError("Invalid input image")

    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive")

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # Scale the image
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    scaled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Create a copy to draw borders on
    result_image = scaled_image.copy()

    # Draw red borders around scaled pixel coordinates
    for x, y in pixel_coords:

        # Validate original coordinates
        if x < 0 or x >= original_width or y < 0 or y >= original_height:
            print(
                f"Warning: Coordinate ({x}, {y}) is outside original image bounds")
            continue

        # Scale the coordinates
        scaled_x = int(x * scale_factor)
        scaled_y = int(y * scale_factor)

        # Calculate rectangle coordinates to represent one scaled pixel
        pixel_size = max(1, int(scale_factor))

        x1 = scaled_x
        y1 = scaled_y
        x2 = min(new_width - 1, scaled_x + pixel_size - 1)
        y2 = min(new_height - 1, scaled_y + pixel_size - 1)

        # Draw red rectangle border around the scaled pixel
        cv2.rectangle(result_image, (x1, y1), (x2, y2),
                      border_color, border_thickness)

    return result_image


def draw_grid(
    cell_size,
    grid_height,
    grid_width,
    background_color=(255, 255, 255),
    line_color=(0, 0, 0),
    line_thickness=1
):
    # Written by Claude

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


def visualize_vector_field(
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

    arrow_size = ceil(cell_size / 8)
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

            color = (0, 0, 255) \
                if (y, x) in input_vector_coords or input_vector_coords == [] \
                else (255, 0, 0)

            draw_arrow(img, start, end, color, arrow_size)

            if scalar_field is not None:
                start_point = (
                    int(start[0]) - int(center_offset/2),
                    int(start[1]) - int(center_offset/2)
                )
                cv2.putText(
                    img, f"{scalar_field[y, x]:.2f}",
                    start_point, 1, 1, (0, 0, 255))

    return img


def draw_arrow(
    img: cv2.Mat,
    start: np.array,
    end: np.array,
    color: np.array,
    size=5
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

    # Get points that form triangle's tip
    front_dir = ((end - start) / np.linalg.norm(end - start))
    side_dir = np.array([front_dir[1], -front_dir[0]])  # 90 deg rot front_dir
    tip = np.array([[end + (3*size) * front_dir], [end + (2*size) * side_dir],
                   [end - (2*size) * side_dir]], dtype=np.int32)

    cv2.fillPoly(img, [tip], color)


def visualize_mask(mask, channel):
    """Helper function to visualize the availability mask for debugging"""
    w, h = mask.shape
    vis_mask = np.zeros((w, h, 4), dtype=np.uint8)
    vis_mask[:, :, 3] = mask.astype(np.uint8) * 255  # Alpha channel
    vis_mask[:, :, channel] = mask.astype(np.uint8) * 255
    return vis_mask


def pixel_map(image, func):
    h, w, _ = image.shape
    mapped = np.zeros_like(image)
    for y in range(h):
        for x in range(w):
            mapped[y, x] = func(image, (x, y))
    return mapped


if __name__ == "__main__":
    img = cv2.imread('data/bases/green_sphere.png')
    scale = 12

    coords_to_highlight = [(10, 7), (11, 7)]

    result = highlight_pixels(img, scale, coords_to_highlight)

    show_scaled('Original', img, scale)
    cv2.imshow('Scaled with Borders', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
