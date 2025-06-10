import cv2
import random
import numpy as np

from vector_field import area_vector
from vector_helpers import quantize_direction
from coloring import mode_color


def pattern_positions(
        mask: np.ndarray,
        pattern_shape: tuple[int, int],
        type: str = "sampling",
        allow_partly_in_mask: bool = False,
        pattern_padding: tuple[int, int] = (0, 0),  # x,y
        density: float = 1,
        max_attempts: int = 1000):

    placements = []
    availability_mask = np.full_like(mask, True, dtype=bool)

    img_height, img_width = mask.shape[:2]
    pattern_height, pattern_width = pattern_shape

    # Find all valid starting positions initially (for faster random selection)
    valid_y, valid_x = np.where(mask)
    valid_positions = list(zip(valid_x, valid_y))

    if not valid_positions:
        print("No valid positions found in boundary")
        return placements

    start_idx = random.randrange(len(valid_positions))

    patterns_placed = 0
    attempts = 0

    positions_to_try = [valid_positions[start_idx]]

    while attempts < max_attempts and valid_positions:
        attempts += 1

        if not positions_to_try:
            # Pick a random valid position
            random_idx = random.randrange(len(valid_positions))
            positions_to_try.append(valid_positions[random_idx])

        x0, y0 = positions_to_try.pop(0)
        x1, y1 = x0 + pattern_width, y0 + pattern_height

        # Extract the region where pattern would be placed
        region_availability = availability_mask[y0:y1, x0:x1]
        region_masked = mask[y0:y1, x0:x1]

        # Check if the pattern fits image at this position
        # and if all pixels in this region aren't already filled
        # and if (all or any) pixels in this region are in original mask
        in_mask = np.any(region_masked) \
            if allow_partly_in_mask else np.all(region_masked)

        if (y1 > img_height or x1 > img_width) \
                or (not np.all(region_availability)) or (not in_mask):
            # Remove this position from consideration
            try:
                idx = valid_positions.index((x0, y0))
                valid_positions.pop(idx)
            except ValueError:
                # ignore it if it wasn't a valid position in the first place
                pass
            continue

        # Add to list of positions
        if random.random() <= density:
            placements.append((x0, y0))
            patterns_placed += 1

        # Update the mask to mark this area as used
        padded_y0, padded_y1 = y0 - pattern_padding[1], y1 + pattern_padding[1]
        padded_x0, padded_x1 = x0 - pattern_padding[0], x1 + pattern_padding[0]
        availability_mask[padded_y0:padded_y1, padded_x0:padded_x1] = False

        # Update valid positions list to remove positions that are no longer valid
        valid_positions = [(x, y) for x, y in valid_positions
                           if availability_mask[y, x]]

        # Compute the next positions to try
        if type == "sampling":
            # Pick a random valid position
            random_idx = random.randrange(len(valid_positions))
            positions_to_try.append(valid_positions[random_idx])
        elif type == "packed":
            # Try all neighbors of current position
            neighbors = [
                # Top row
                (padded_x0 - pattern_width, padded_y0 - pattern_height),  # NW
                (padded_x0,                 padded_y0 - pattern_height),  # N
                (padded_x0 + pattern_width, padded_y0 - pattern_height),  # NE

                # Middle row (left and right of current rect)
                (padded_x0 - pattern_width, padded_y0),           # W
                (padded_x0 + pattern_width, padded_y0),           # E

                # Bottom row
                (padded_x0 - pattern_width, padded_y0 + pattern_height),  # SW
                (padded_x0,                 padded_y0 + pattern_height),  # S
                (padded_x0 + pattern_width, padded_y0 + pattern_height),  # SE
            ]
            # Remove impossible neighbors
            neighbors = [(x, y) for x, y in neighbors if x >= 0 and y >= 0]

            positions_to_try += neighbors
        else:
            raise ValueError("Type must be 'sampling' or 'packed'")

    print(
        f"Placed {patterns_placed} patterns" + f"in {attempts} attempts, " +
        f"remaining valid positions: {bool(valid_positions)}")
    return placements


def find_pattern(
    vector_field: np.ndarray,
    point: tuple[int, int],
    patterns: np.ndarray
):
    x, y = point
    rows, cols, h, w, _ = patterns.shape

    if rows != 3 or cols != 3:
        raise ValueError("Patterns should be within a 3x3 matrix")

    # Find the direction of that area
    area_dir = area_vector(vector_field, (y, x, w, h))

    # Choose appropriate pattern according to direction
    pattern_dir_x, pattern_dir_y = quantize_direction(area_dir)

    # Deal with placements without a direction
    if pattern_dir_y == 0 and pattern_dir_x == 0:
        raise ValueError(f"Point {point} has no direction in vector field")

    return patterns[pattern_dir_x+1, pattern_dir_y+1]


def split_oriented_spritesheet(img: cv2.Mat):
    """
    Splits a 3x3 spritesheet into individual sprites based on orientation.

    The spritesheet is assumed to have 9 sprites arranged in a 3x3 grid.
    Each sprite represents a different orientation:
        [0,0]: top-left     [0,1]: top       [0,2]: top-right
        [1,0]: left         [1,1]: center    [1,2]: right
        [2,0]: bottom-left  [2,1]: bottom    [2,2]: bottom-right

    Args:
        img: OpenCV image (H,W,C) format where H = sprite_h*3 and W = sprite_w*3

    Returns:
        sprites: Numpy array of shape (3,3,sprite_h,sprite_w,4) 
                 containing all sprites indexed as sprites[row, col]
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
            # This section has an apparent x, y swap. This is because we need to
            # access the image with OpenCV's [y, x] indexing, but the sprites
            # themselves are laid out in the more intuitive [x, y] indexing.

            # Calculate starting pixel coordinates in the original image
            # Convert grid position to pixel position
            start_x = (row_idx) * sprite_height
            start_y = (col_idx) * sprite_width

            # Extract the sprite and store it in our output array
            sprites[row_idx, col_idx, :, :, :] = img[start_y:start_y+sprite_height,
                                                     start_x:start_x+sprite_width, :]

    return sprites


def place_pattern(
        destination: cv2.Mat,
        pattern: cv2.Mat,
        point: tuple[int, int],
        mask: np.ndarray,
        color_map: cv2.Mat | None = None,
        type: str = "region"
):

    x0, y0 = point

    # Boundary check
    h, w = pattern.shape[:2]
    if y0 + h > destination.shape[0] or x0 + w > destination.shape[1]:
        return False

    # Find pattern color

    if type == "region":
        region_of_interest = color_map[y0: y0 + h, x0: x0 + w, :]
        region_color = mode_color(region_of_interest)

    # Placing

    for x in range(x0, x0 + w):
        for y in range(y0, y0 + h):
            if not mask[y, x]:
                continue

            alpha = pattern[y-y0, x-x0, 3]

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
