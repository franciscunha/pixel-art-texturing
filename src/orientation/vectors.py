import numpy as np


def average_vector(vectors: list[np.array]):
    """Calculate the average of a list of vectors."""
    sum_vec = np.array((0, 0), dtype=np.float64)
    count = 0
    for vector in vectors:
        if np.all(vector == 0):
            continue
        sum_vec += vector
        count += 1
    return sum_vec / count if count != 0 else sum_vec


def area_vector(
    vector_field: np.ndarray,
    region: tuple[int, int, int, int]
) -> np.ndarray:
    """
    Extract and average vectors from a rectangular region in a vector field.

    Args:
        vector_field: 3D array of shape (height, width, 2) representing a 2D field of vectors.
        region: Tuple (y, x, height, width) defining the rectangular region to process.
            y, x: Top-left coordinates of the region.
            height, width: Dimensions of the region.

    Returns:
        np.ndarray: A single 2D vector representing the average of all vectors in the region.
    """
    y, x, height, width = region
    # Extract the vectors in the specified rectangular region
    area = vector_field[y:y+height, x:x+width, :]
    # Flatten the 2D grid of vectors into a 1D array of vectors, then average
    return average_vector(area.reshape(height * width, 2))


def compress_vector_field(
    vector_field: np.ndarray,
    window_size: tuple[int, int]
) -> np.ndarray:
    """
    Compress a vector field by averaging vectors within non-overlapping windows.

    Args:
        vector_field: 3D array of shape (height, width, 2) representing a 2D field of vectors.
        window_size: Tuple (window_height, window_width) defining the size of each window.
            The vector field dimensions must be evenly divisible by these values.

    Returns:
        np.ndarray: Compressed vector field of shape (height/window_height, width/window_width, 2).

    Raises:
        ValueError: If the vector field dimensions are not evenly divisible by the window size.
    """
    window_height, window_width = window_size
    height, width, _ = vector_field.shape

    # Skip if no compression is asked for
    if window_height == 1 and window_width == 1:
        return vector_field

    # Check if the vector field can be evenly divided into windows
    if height % window_height != 0 or width % window_width != 0:
        raise ValueError(
            "Vector field dimensions must be evenly divisible by the window size"
        )

    # Calculate dimensions of the compressed field
    compressed_height = height // window_height
    compressed_width = width // window_width

    # Initialize the compressed vector field
    compressed = np.zeros((compressed_height, compressed_width, 2))

    # Process each window
    for y in range(0, height, window_height):
        for x in range(0, width, window_width):
            # Calculate indices in the compressed field
            compressed_y = y // window_height
            compressed_x = x // window_width

            # Calculate the average vector for this window
            compressed[compressed_y, compressed_x] = area_vector(
                vector_field,
                (y, x, window_height, window_width)
            )

    return compressed


def quantize_direction(direction: np.ndarray) -> np.ndarray:
    """
    Find the closest cardinal or diagonal direction to the input vector.

    This function quantizes a 2D vector to the nearest of 8 possible directions:
    - 4 cardinal directions: up, right, down, left (0,1), (1,0), (0,-1), (-1,0)
    - 4 diagonal directions: (±0.7071, ±0.7071) which are normalized versions of (±1, ±1)

    Args:
        direction: A 2D vector (numpy array of shape (2,)) representing a direction.

    Returns:
        np.ndarray: An integer vector (either cardinal or diagonal) representing 
        the closest standard direction. Returns [0, 0] if no valid direction is found.

    Notes:
        - Directions are matched based on the dot product (cosine similarity).
        - For normalized input vectors, the dot product is maximized (closer to 1)
          when the directions are similar.
        - The function returns integer coordinates (-1, 0, or 1 for each component).
    """
    # Define the 8 standard directions (all normalized to unit length)
    standard_directions = [
        np.array([-0.7071, -0.7071]),  # Northwest
        np.array([-1, 0]),             # West
        np.array([-0.7071, 0.7071]),   # Southwest
        np.array([0, -1]),             # North
        np.array([0, 1]),              # South
        np.array([0.7071, -0.7071]),   # Northeast
        np.array([1, 0]),              # East
        np.array([0.7071, 0.7071])     # Southeast
    ]

    # We know this to be a lower bound for similarity, so we can skip a few comparisons
    # Cannot take higher lower bound because no guarantee vector is unitary
    # (dot = 0 if two vectors are 90 degrees apart)
    best_similarity = 0
    best_direction = None

    for candidate in standard_directions:
        # Calculate dot product as a measure of similarity
        similarity = np.dot(candidate, direction)

        # If directions are nearly identical (accounting for floating point errors)
        if abs(similarity - 1) < 1e-4:
            best_direction = candidate
            break

        # Update best match if this similarity is higher
        if similarity > best_similarity:
            best_similarity = similarity
            best_direction = candidate

    # If no valid direction found, return zero vector
    if best_direction is None:
        return np.array([0, 0])

    # Convert the floating-point direction to integers (-1, 0, or 1)
    return np.round(best_direction).astype(np.int8)
