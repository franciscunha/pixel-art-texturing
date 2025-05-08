import random
import cv2
import numpy as np

from annotations import areaVector, compressVectorField, drawOnImage, parseCurves, visualizeVectorField
from random_placement import placePattern, showScaled


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
    dir_h, dir_w, pattern_height, pattern_width, _ = patterns.shape

    if dir_w != 3 or dir_h != 3:
        raise ValueError("Patterns should be within a 3x3 matrix")

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
            areaVector(vector_field, (y0, x0, pattern_width, pattern_height))

        # Choose appropriate pattern according to direction
        pattern_dir_y, pattern_dir_x = findClosestDirection(area_dir)
        pattern = patterns[pattern_dir_y+1, pattern_dir_x+1]

        # Ignore placements without a direction
        if pattern_dir_y == 0 and pattern_dir_x == 0:
            continue

        # Place the pattern
        success = placePattern(result, pattern, y0, x0, hsv_shift)

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

        # print(f"Placed pattern #{patterns_placed} direction {(pattern_dir_x, pattern_dir_y)} at ({x0}, {y0})")

    print(
        f"Placed {patterns_placed} patterns out of {num_patterns} requested (in {attempts} attempts)")
    return result


def splitOrientedSpritesheet(img: cv2.Mat):
    """
    Splits a 3x3 spritesheet into individual sprites based on orientation.

    The spritesheet is assumed to have 9 sprites arranged in a 3x3 grid.
    Each sprite represents a different orientation:
        [0,0]: top-left     [0,1]: top       [0,2]: top-right
        [1,0]: left         [1,1]: center    [1,2]: right
        [2,0]: bottom-left  [2,1]: bottom    [2,2]: bottom-right

    Args:
        img: OpenCV image (H,W,C) format where H = sprite_height*3 and W = sprite_width*3

    Returns:
        sprites: Numpy array of shape (3,3,sprite_height,sprite_width,4) containing all sprites
                 indexed as sprites[row, col]
    """
    # Get total image dimensions
    img_height, img_width, _ = img.shape

    # Calculate individual sprite dimensions
    sprite_height = int(img_height / 3)
    sprite_width = int(img_width / 3)

    # Initialize output array: [row, col, sprite_height, sprite_width, channels]
    sprites = np.empty((3, 3, sprite_height, sprite_width, 4))

    # Iterate through the 3x3 grid
    for row_idx in range(3):
        for col_idx in range(3):
            # This section has an apparent x, x swap. This is because we need to access the image
            # with OpenCV's [y, x] indexing, but the sprites themselves are laid out in the more
            # intuitive [x, y] indexing.

            # Calculate starting pixel coordinates in the original image
            # Convert grid position to pixel position
            start_x = (row_idx) * sprite_height
            start_y = (col_idx) * sprite_width

            # Extract the sprite and store it in our output array
            sprites[row_idx, col_idx, :, :, :] = img[start_y:start_y+sprite_height,
                                                     start_x:start_x+sprite_width, :]

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
