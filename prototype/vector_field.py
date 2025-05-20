import numpy as np

from vector_helpers import average_vector


def divergence(vector_field: np.ndarray):
    return np.ufunc.reduce(np.add, [np.gradient(vector_field[:, :, i], axis=i) for i in range(2)])


def compress_vector_field(vector_field: np.ndarray, window_size: tuple[int, int]) -> np.ndarray:
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


def area_vector(vector_field: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
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
