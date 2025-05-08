import cv2
import numpy as np


# Written by Claude
def drawGrid(cell_size, grid_height, grid_width, background_color=(255, 255, 255), line_color=(0, 0, 0), line_thickness=1):
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


def visualizeVectorField(vec_field: np.array):
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
    img = drawGrid(cell_size, h, w)

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

            drawArrow(img, start, end, (255, 0, 0), 3)

    return img


def drawArrow(img: cv2.Mat, start: np.array, end: np.array, color: np.array, size=5):
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


def drawOnImage(img: cv2.Mat, scale: int):
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
                drawArrow(img, points[-2], points[-1], (255, 0, 0))
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


def parseCurve(points: np.array):
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


def avgVector(vectors: list[np.array]):
    """Calculate the average of a list of vectors."""
    sum_vec = np.array((0, 0), dtype=np.float64)
    count = 0
    for vector in vectors:
        if np.all(vector == 0):
            continue
        sum_vec += vector
        count += 1
    return sum_vec / count if count != 0 else sum_vec


def parseCurves(curves: list[np.array], img_h: int, img_w: int):
    """
    Parse multiple curves to create a vector field.
    Assumes curves contain points in (x,y) format.
    Returns a vector field of shape (h, w, 2) where each vector is (dx, dy).
    """
    all_influences = {}

    for curve in curves:
        curve_influences = parseCurve(curve)

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
            vector_field[y, x] = avgVector(vectors)

    return vector_field


def areaVector(vector_field: np.array, rect: tuple[int, int, int, int]):
    y, x, h, w = rect
    area = vector_field[y:y+h, x:x+w, :]
    return avgVector(area.reshape(h * w, 2))


def compressVectorField(vector_field: np.array, window_size: tuple[int, int]):
    window_h, window_w = window_size
    h, w, _ = vector_field.shape

    if h % window_h != 0 or w % window_w != 0:
        raise ValueError(
            "Vector field resolution must be evenly divisible by the window size")

    compressed = np.zeros((int(h / window_h), int(w / window_w), 2))

    for y in range(0, h, window_h):
        for x in range(0, w, window_w):
            compressed[int(y/window_h), int(x/window_w)] =\
                areaVector(vector_field, (y, x, window_h, window_w))

    return compressed


def main():
    # shape = cv2.imread("data/shaded_tree.png", cv2.IMREAD_UNCHANGED)

    # if shape is None:
    #     raise FileNotFoundError()

    shape = (32, 32)
    scale = 2

    canvas = np.zeros((shape[0], shape[1], 3), np.uint8)

    curves = drawOnImage(canvas, scale)
    # influences = parseCurves(curves, shape[0]*scale, shape[1]*scale)
    influences = compressVectorField(
        parseCurves(curves, shape[0]*scale, shape[1]*scale), (scale, scale)
    )

    vector_field_img = visualizeVectorField(influences)

    cv2.imshow("Vector field", vector_field_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
