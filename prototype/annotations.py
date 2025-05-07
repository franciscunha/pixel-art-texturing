import cv2
import numpy as np


def parseCurve(points: list[np.array]):
    pass


def drawArrow(img: cv2.Mat, start: np.array, end: np.array, color: np.array):
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("Wrong input shape")

    cv2.line(img, start, end, color, 5)

    # Get points that form triangle's tip
    front_dir = ((end - start) / np.linalg.norm(end - start))
    side_dir = np.array([front_dir[1], -front_dir[0]])  # 90 deg rot front_dir
    tip = np.array([[end + 15 * front_dir], [end + 10 * side_dir],
                   [end - 10 * side_dir]], dtype=np.int32)
    print(
        f"start: {start}, end: {end}, front dir: {front_dir}, side dir: {side_dir}, points: {tip}")

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

    canvas = np.zeros((64, 64, 3), np.uint8)

    drawOnImage(canvas, 8)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
