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
    influences = {}
    for i in range(len(points) - 1):
        vec = points[i+1] - points[i]
        vec = vec / np.linalg.norm(vec)

        key = (points[i][0], points[i][1])

        if key not in influences:
            influences[key] = []
        influences[key].append(vec)

    return influences


def avgVector(vectors: list[np.array]):
    sum_vec = np.array((0, 0), dtype=np.float64)
    for vector in vectors:
        sum_vec += vector
    return sum_vec / len(vectors)


def parseCurves(curves: list[np.array], img_h: int, img_w: int):
    all_influences = {}

    for curve in curves:
        curve_influences = parseCurve(curve)

        for point, vectors in curve_influences.items():
            if point in all_influences:
                all_influences[point] = all_influences[point] + vectors
            else:
                all_influences[point] = vectors

    # At each point, take the average of all influences
    influences = np.zeros((img_h, img_w, 2), dtype=np.float64)
    for point, vectors in all_influences.items():
        influences[point[0], point[1]] = avgVector(vectors)

    return influences


def main():
    # shape = cv2.imread("data/shaded_tree.png", cv2.IMREAD_UNCHANGED)

    # if shape is None:
    #     raise FileNotFoundError()

    canvas = np.zeros((16, 16, 3), np.uint8)

    curves = drawOnImage(canvas, 8)
    influences = parseCurves(curves, 16*8, 16*8)

    vector_field_img = visualizeVectorField(influences)

    cv2.imshow("Vector field", vector_field_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
