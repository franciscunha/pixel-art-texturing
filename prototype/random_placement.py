import cv2
import numpy as np
import random

from placement import place_pattern
from visualizations import show_scaled


def place_random_singular_pattern_within_boundary(
        destination: cv2.Mat,
        pattern: cv2.Mat,
        boundary: cv2.Mat,
        pattern_padding=0,
        num_patterns=20,
        max_attempts=1000,
        hsv_shift: tuple[int, int, int] | None = None):
    """
    Place random patterns within the boundary using a mask to track available space.

    Args:
        destination: The destination image where patterns will be placed
        pattern: The pattern to place
        boundary_img: The image defining the boundary
        pattern_padding: Space between any two placed patterns, in pixels
        num_patterns: Number of patterns to try to place
        max_attempts: Maximum attempts to find valid positions

    Returns: The modified destination with patterns placed
    """
    result = destination.copy()
    pattern_height, pattern_width = pattern.shape[:2]
    height, width = boundary.shape[:2]

    # Create a mask where True means space is available (initial mask is the boundary)
    # We start with the boundary where alpha = 255 means available space
    availability_mask = boundary[:, :, 3] == 255

    # Find all valid starting positions initially (for faster random selection)
    valid_y, valid_x = np.where(availability_mask)
    valid_positions = list(zip(valid_y, valid_x))

    if not valid_positions:
        print("No valid positions found in boundary")
        return result

    patterns_placed = 0
    attempts = 0

    while patterns_placed < num_patterns and attempts < max_attempts and valid_positions:
        attempts += 1

        # Pick a random valid position
        position_idx = random.randrange(len(valid_positions))
        y0, x0 = valid_positions[position_idx]
        y1, x1 = y0 + pattern_height, x0 + pattern_width

        # Check if the pattern would fit at this position
        if (y1 > height or x1 > width):
            # Remove this position from consideration
            valid_positions.pop(position_idx)
            continue

        # Extract the region where pattern would be placed
        region = availability_mask[y0:y1, x0:x1]

        # Check if all pixels in this region are available (True in mask)
        if not np.all(region):
            # Remove this position from consideration
            valid_positions.pop(position_idx)
            continue

        # Place the pattern
        success = place_pattern(result, pattern, y0, x0, hsv_shift)

        if not success:
            continue

        patterns_placed += 1

        # Update the mask to mark this area as used
        padded_y0, padded_y1 = y0 - pattern_padding, y1 + pattern_padding
        padded_x0, padded_x1 = x0 - pattern_padding, x1 + pattern_padding
        availability_mask[padded_y0:padded_y1, padded_x0:padded_x1] = False

        # Update valid positions list to remove positions that are no longer valid
        valid_positions = [(y, x)
                           for y, x in valid_positions if availability_mask[y, x]]

        print(f"Placed pattern {patterns_placed} at ({y0}, {x0})")

    print(
        f"Placed {patterns_placed} patterns out of {num_patterns} requested (in {attempts} attempts)")
    return result


def main():
    shape = cv2.imread("data/shaded_tree.png", cv2.IMREAD_UNCHANGED)
    pattern = cv2.imread(f"data/leaf2_pattern.png", cv2.IMREAD_UNCHANGED)
    boundary = cv2.imread(
        f"data/shaded_tree_canopy_full_boundary.png", cv2.IMREAD_UNCHANGED)

    if shape is None or pattern is None or boundary is None:
        raise FileNotFoundError()

    show_scaled("Input", shape, 4)
    # If this comes after a call to placePattern with hsv_shift, it'll just look like a square
    show_scaled("Pattern", pattern, 64)

    result = place_random_singular_pattern_within_boundary(
        shape, pattern, boundary, 2, 60, hsv_shift=(0, 0, 30))

    show_scaled("Output", result, 4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
