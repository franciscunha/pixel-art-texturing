import cv2
import numpy as np


# Written by Claude
def draw_grid(cell_size, grid_height, grid_width, background_color=(255, 255, 255), line_color=(0, 0, 0), line_thickness=1):
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


def visualize_vector_field(vec_field: np.array):
    """
    Visualize a vector field. The input array should have shape (h, w, 2),
    where vec_field[y, x] contains the (dx, dy) vector at position (x, y).
    """
    h, w, vec_shape = vec_field.shape
    if vec_shape != 2:
        raise ValueError("Expected shape (h, w, 2)")

    cell_size = 24
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

            draw_arrow(img, start, end, (255, 0, 0), 3)

    return img


def draw_arrow(img: cv2.Mat, start: np.array, end: np.array, color: np.array, size=5):
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


def draw_on_image(img: cv2.Mat, scale: int):
    # Rescale image
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)

    # Creating a window
    window_name = 'Annotation'
    cv2.namedWindow(window_name)

    curves = []
    points = []

    # Mouse callback function

    def callback(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(np.array((x, y)))
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            points.append(np.array((x, y)))
            cv2.line(img, points[-2], points[-1], (255, 0, 0), 5)
        elif event == cv2.EVENT_LBUTTONUP:
            if len(points) > 1:
                draw_arrow(img, points[-2], points[-1], (255, 0, 0))
                curves.append(np.array(points))
            points.clear()

    # Bind the callback function to window
    cv2.setMouseCallback(window_name, callback)

    # Show drawing. Note this works like a game engine's Draw() callback
    while True:
        cv2.imshow(window_name, img)
        if cv2.waitKey(20) == 27:
            break

    return curves


def parse_curve(points: np.array):
    """
    Parse a curve to extract influence vectors.
    Assumes points are in (x,y) format.
    Returns a dictionary with (x,y) tuple keys and lists of influence vectors.
    """
    influences = {}

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]

        # Calculate the vector (dx, dy) between consecutive points
        vec = p2 - p1

        # Normalize the vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Store as (x,y) for consistency
        key = (int(p1[0]), int(p1[1]))

        if key not in influences:
            influences[key] = []

        influences[key].append(vec)

    return influences


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


def parse_curves(curves: list[np.array], img_h: int, img_w: int):
    """
    Parse multiple curves to create a vector field.
    Assumes curves contain points in (x,y) format.
    Returns a vector field of shape (h, w, 2) where each vector is (dx, dy).
    """
    all_influences = {}

    for curve in curves:
        curve_influences = parse_curve(curve)

        for point, vectors in curve_influences.items():
            if point in all_influences:
                all_influences[point] = all_influences[point] + vectors
            else:
                all_influences[point] = vectors

    # Initialize the vector field with zeros
    # Note: The array is indexed as [y, x] but contains (dx, dy) vectors
    vector_field = np.zeros((img_h, img_w, 2), dtype=np.float64)

    # At each point, take the average of all influences
    for point, vectors in all_influences.items():
        # Convert (x,y) point to array indices [y,x]
        x, y = point

        # Check if the point is within array bounds
        if 0 <= y < img_h and 0 <= x < img_w:
            vector_field[y, x] = average_vector(vectors)

    return vector_field


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


def main():
    # shape = cv2.imread("data/shaded_tree.png", cv2.IMREAD_UNCHANGED)

    # if shape is None:
    #     raise FileNotFoundError()

    shape = (32, 32)
    scale = 2

    canvas = np.zeros((shape[0], shape[1], 3), np.uint8)

    curves = draw_on_image(canvas, scale)
    # influences = parseCurves(curves, shape[0]*scale, shape[1]*scale)
    influences = compress_vector_field(
        parse_curves(curves, shape[0]*scale, shape[1]*scale), (scale, scale)
    )

    vector_field_img = visualize_vector_field(influences)

    cv2.imshow("Vector field", vector_field_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
