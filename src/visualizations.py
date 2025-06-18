from math import ceil
import cv2
import numpy as np

from src.orientation.drawing import draw_vector_field
from src.orientation.vectors import compress_vector_field


# Scaling helpers

def show_scaled(title: str, img: cv2.Mat, factor: int):
    scaled = cv2.resize(img.copy(), dsize=None, fx=factor,
                        fy=factor, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(title, scaled)


def save_scaled(filepath: str, img: cv2.Mat, factor: int):
    scaled = cv2.resize(img.copy(), dsize=None, fx=factor,
                        fy=factor, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filepath, scaled)


# Visualize processed data

def visualize_vector_field(vector_field, annotations, grid_scale, cell_size):
    annotations_compressed = compress_vector_field(annotations, grid_scale)
    annotated_coords = [
        (y, x)
        for y, x in np.ndindex(annotations_compressed.shape[:2])
        if np.any(annotations_compressed[y, x] != 0)
    ]

    return draw_vector_field(
        compress_vector_field(vector_field, grid_scale),
        input_vector_coords=annotated_coords,
        cell_size=cell_size
    )


def visualize_annotations(annotations, grid_scale, cell_size):
    return draw_vector_field(
        compress_vector_field(annotations, grid_scale),
        cell_size=cell_size
    )


def visualize_positions(source, positions):
    red = np.array([0, 0, 255, 255])

    result = source.copy()
    for position in positions:
        x, y = position

        # Draw a red pixel at center of where element would be
        result[y+1, x+1] = red

    return result


# Create images

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


def visualize_mask(mask, channel):
    """Helper function to visualize the availability mask for debugging"""
    w, h = mask.shape
    vis_mask = np.zeros((w, h, 4), dtype=np.uint8)
    vis_mask[:, :, 3] = mask.astype(np.uint8) * 255  # Alpha channel
    vis_mask[:, :, channel] = mask.astype(np.uint8) * 255
    return vis_mask


# Modify images

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
