import cv2
import numpy as np

from src.orientation.vectors import average_vector
from src.orientation.drawing import draw_arrow


# Get user input

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


# Parse user input

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
