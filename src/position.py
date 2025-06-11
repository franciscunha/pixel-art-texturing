
import random

import numpy as np


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

    while attempts < max_attempts and (valid_positions or positions_to_try):
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
