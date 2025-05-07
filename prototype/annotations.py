import cv2
import numpy as np


def parseCurve(points: list[np.array]):
    pass


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
    h, w, vec_shape = vec_field.shape
    if vec_shape != 2:
        raise ValueError("Expected shape (h, w, 2)")

    cell_size = 24
    center_offset = cell_size / 2
    # img = np.zeros((h * cell_size, w * cell_size, 4), np.uint8)
    img = drawGrid(cell_size, h, w)

    for y in range(h):
        for x in range(w):
            vec = vec_field[y, x]

            if vec.all() == 0:
                continue

            start = np.array([(y * cell_size) + center_offset,
                              (x * cell_size) + center_offset], dtype=np.uint16)
            end = np.array((vec * center_offset) + start, dtype=np.uint16)

            drawArrow(img, start, end, (255, 0, 0), 3)

    return img


def drawArrow(img: cv2.Mat, start: np.array, end: np.array, color: np.array, size=5):
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("Wrong input shape")

    cv2.line(img, start, end, color, size)

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

    # Mouse callback function
    points = []

    def callback(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            points.append((x, y))
            cv2.line(img, points[-2], points[-1], (255, 0, 0), 5)
        elif event == cv2.EVENT_LBUTTONUP:
            if len(points) > 1:
                drawArrow(img, np.array(points[-2]),
                          np.array(points[-1]), (255, 0, 0))
            points.clear()
            # At this point, we can create a vector

    # Bind the callback function to window
    cv2.setMouseCallback(window_name, callback)

    # Show drawing. Note this works like a game engine's Draw() callback
    while True:
        cv2.imshow(window_name, img)
        if cv2.waitKey(20) == 27:
            break


def main():
    # shape = cv2.imread("data/shaded_tree.png", cv2.IMREAD_UNCHANGED)

    # if shape is None:
    #     raise FileNotFoundError()

    # canvas = np.zeros((64, 64, 3), np.uint8)

    # drawOnImage(canvas, 8)

    # Create empty array of shape (16, 16, 2)
    sparse_vectors = np.zeros((16, 16, 2))

    # Hardcoded positions for non-zero vectors
    # Each entry is a tuple of (row, column, vector)
    # where vector is already normalized
    vector_positions = [
        # Normalized vector pointing up-right
        (1, 3, np.array([0.6, 0.8])),
        (2, 7, np.array([0.0, 1.0])),         # Normalized vector pointing up
        # Normalized vector pointing right
        (4, 12, np.array([1.0, 0.0])),
        # Normalized vector pointing down-right (45 degrees)
        (5, 5, np.array([0.7071, -0.7071])),
        # Normalized vector pointing down-left
        (6, 2, np.array([-0.5, -0.866])),
        (7, 8, np.array([-1.0, 0.0])),        # Normalized vector pointing left
        # Normalized vector (approximately)
        (9, 10, np.array([0.9487, 0.3162])),
        # Normalized vector (approximately)
        (11, 1, np.array([0.2425, 0.9701])),
        # Normalized vector (approximately)
        (12, 12, np.array([-0.9636, 0.2673])),
        # Normalized vector (approximately)
        (14, 9, np.array([-0.3015, -0.9535]))
    ]

    # Place the vectors at specified positions
    for row, col, vector in vector_positions:
        sparse_vectors[row, col] = vector

    img = visualizeVectorField(sparse_vectors)
    cv2.imshow("Vec field", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
