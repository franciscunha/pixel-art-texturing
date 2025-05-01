import cv2
import numpy as np
import random


def visualizeMask(mask, original_shape):
    """Helper function to visualize the availability mask for debugging"""
    vis_mask = np.zeros_like(original_shape)
    vis_mask[:, :, 3] = mask.astype(np.uint8) * 255  # Alpha channel
    vis_mask[:, :, 0] = mask.astype(np.uint8) * 255  # Blue
    vis_mask[:, :, 1] = 0  # Green
    vis_mask[:, :, 2] = 0  # Red
    return vis_mask


def placeRandomPatternsWithinBoundary(destination: cv2.Mat, pattern: cv2.Mat, boundary: cv2.Mat, pattern_padding=0, num_patterns=20, max_attempts=1000):
    """
    Place random patterns within the boundary using a mask to track available space.

    Args:
        destination: The destination image where patterns will be placed
        pattern: The pattern to place
        boundary_img: The image defining the boundary
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
    valid_positions = list(zip(valid_x, valid_y))

    if not valid_positions:
        print("No valid positions found in boundary")
        return result

    patterns_placed = 0
    attempts = 0

    while patterns_placed < num_patterns and attempts < max_attempts and valid_positions:
        attempts += 1

        # Pick a random valid position
        position_idx = random.randrange(len(valid_positions))
        x0, y0 = valid_positions[position_idx]
        x1, y1 = x0 + pattern_width, y0 + pattern_height

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
        success = placePattern(result, pattern, x0, y0)

        if not success:
            continue

        patterns_placed += 1

        # Update the mask to mark this area as used
        padded_y0, padded_y1 = y0 - pattern_padding, y1 + pattern_padding
        padded_x0, padded_x1 = x0 - pattern_padding, x1 + pattern_padding
        availability_mask[padded_y0:padded_y1, padded_x0:padded_x1] = False

        # Update valid positions list to remove positions that are no longer valid
        valid_positions = [(x, y)
                           for x, y in valid_positions if availability_mask[y, x]]

        print(f"Placed pattern {patterns_placed} at ({x0}, {y0})")

    print(
        f"Placed {patterns_placed} patterns out of {num_patterns} requested (in {attempts} attempts)")
    return result


def showScaled(title: str, img: cv2.Mat, factor: int):
    scaled = cv2.resize(img, dsize=None, fx=factor,
                        fy=factor, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(title, scaled)


def placePattern(destination: cv2.Mat, pattern: cv2.Mat, x: int, y: int):
    # Boundary check
    h, w = pattern.shape[:2]
    if y + h > destination.shape[0] or x + w > destination.shape[1]:
        # TODO better handling
        return False

    # Extract alpha
    pattern_alpha = pattern[:, :, 3] / 255.0
    pattern_bgr = pattern[:, :, :3]  # note the numpy notation

    # Placing
    region_of_interest = destination[y: y + h, x: x + w, :3]
    for c in range(3):
        # implicit double for loop due to numpy
        destination[y: y + h, x: x + w, c] = \
            (1 - pattern_alpha) * region_of_interest[:, :, c] \
            + pattern_alpha * pattern_bgr[:, :, c]

    return True


def main():
    shape = cv2.imread("data/tree_shape.png", cv2.IMREAD_UNCHANGED)
    pattern = cv2.imread("data/leaf_pattern.png", cv2.IMREAD_UNCHANGED)
    boundary = cv2.imread("data/tree_boundary.png", cv2.IMREAD_UNCHANGED)

    if shape is None or pattern is None or boundary is None:
        raise FileNotFoundError()

    result = placeRandomPatternsWithinBoundary(shape, pattern, boundary, 1, 30)

    showScaled("Shape", shape, 8)
    showScaled("Pattern", pattern, 16)
    showScaled("Result", result, 8)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
