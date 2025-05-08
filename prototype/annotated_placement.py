import random
import cv2
import numpy as np

from annotations import areaVector, compressVectorField, drawOnImage, parseCurves, visualizeVectorField
from random_placement import placePattern, showScaled


# TODO deal with direction of region being 0

def findClosestDirection(dir: np.array):
    possible_dirs = [np.array([-0.7071, -0.7071]), np.array([-1, 0]), np.array([-0.7071, +0.7071]),
                     np.array([0, -1]), np.array([0, 1]),
                     np.array([+0.7071, -0.7071]), np.array([+1, 0]), np.array([+0.7071, +0.7071])]

    best_dot, best_candidate = 0, None

    for candidate in possible_dirs:
        dot = np.dot(candidate, dir)

        if abs(dot - 1) < 1e-4:
            best_candidate = candidate
            break

        if dot > best_dot:
            best_dot = dot
            best_candidate = candidate

    if best_candidate is None:
        return np.array([0, 0])

    return np.round(best_candidate).astype(np.int8)


def placePatternsWithinBoundary(
        destination: cv2.Mat,
        patterns: np.array,
        boundary: cv2.Mat,
        vector_field: np.array,
        pattern_padding=0,
        num_patterns=20,
        max_attempts=1000,
        hsv_shift: tuple[int, int, int] | None = None):
    """
    Place random patterns within the boundary using a mask to track available space.

    Args:
        destination: The destination image where patterns will be placed
        patterns: A 3x3 matrix where each cell is a pattern aligned with its coordinates direction
        boundary: The image defining the boundary
        vector_field: Matrix of same size as image, defining a direction per pixel
        pattern_padding: Space between any two placed patterns, in pixels
        num_patterns: Number of patterns to try to place
        max_attempts: Maximum attempts to find valid positions
        hsv_shift: How to change the background's color to give a color to the placed pattern

    Returns: The modified destination with patterns placed
    """

    result = destination.copy()
    boundary_height, boundary_width = boundary.shape[:2]
    dir_w, dir_h, pattern_height, pattern_width, _ = patterns.shape

    if dir_w != 3 or dir_h != 3:
        raise ValueError("Patterns should be within a 3x3 matrix")

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
        if (y1 > boundary_height or x1 > boundary_width):
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

        # Find the direction of that area
        area_dir =\
            areaVector(vector_field, (x0, y0, pattern_width, pattern_height))

        # Choose appropriate pattern according to direction
        pattern_dir_x, pattern_dir_y = findClosestDirection(area_dir)
        pattern = patterns[pattern_dir_x+1, pattern_dir_y+1]

        # Ignore placements without a direction
        if pattern_dir_x == 0 and pattern_dir_y == 0:
            continue

        # Place the pattern
        success = placePattern(result, pattern, x0, y0, hsv_shift)

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

        # print(f"Placed pattern #{patterns_placed} direction {(pattern_dir_x, pattern_dir_y)} at ({x0}, {y0})")

    print(
        f"Placed {patterns_placed} patterns out of {num_patterns} requested (in {attempts} attempts)")
    return result


def splitOrientedSpritesheet(img: cv2.Mat):
    img_h, img_w, _ = img.shape
    h, w = int(img_h / 3), int(img_w / 3)

    sprites = np.empty((3, 3, h, w, 4))
    for j in range(-1, 2):
        for i in range(-1, 2):
            y, x = (j+1) * h, (i+1) * w
            sprites[i+1, j+1, :, :, :] = img[y:y+h, x:x+w, :]

    return sprites


def main():

    base = cv2.imread("data/shaded_tree.png", cv2.IMREAD_UNCHANGED)
    pattern_sheet = cv2.imread(
        f"data/slynrd_leaf_spritesheet.png", cv2.IMREAD_UNCHANGED)
    boundary = cv2.imread(
        f"data/shaded_tree_canopy_full_boundary.png", cv2.IMREAD_UNCHANGED)

    if base is None or pattern_sheet is None or boundary is None:
        raise FileNotFoundError()

    scale = 4
    shape = base.shape[:2]
    patterns = splitOrientedSpritesheet(pattern_sheet)

    curves = drawOnImage(base, scale)
    influences = compressVectorField(
        parseCurves(curves, shape[0]*scale, shape[1]*scale), (scale, scale)
    )

    vector_field_img = visualizeVectorField(
        compressVectorField(influences, (4, 4)))
    cv2.imshow("Vector field", vector_field_img)

    result = placePatternsWithinBoundary(base, patterns, boundary, influences,
                                         pattern_padding=1, num_patterns=50, hsv_shift=(0, 0, -20))

    showScaled("Output", result, scale)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
