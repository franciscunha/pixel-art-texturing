import cv2
import numpy as np

from visualizations import show_scaled, visualize_mask


def mask_from_boundary(boundary: np.ndarray):
    # We start with the boundary where alpha = 255 means available space
    return boundary[:, :, 3] == 255


def pad_mask(mask: np.ndarray, padding_size: int):
    h, w = mask.shape
    p = padding_size
    if p <= 0:
        return mask

    padded = mask.copy()

    for y in range(p, h - p):
        for x in range(p, w - p):
            window = mask[y-p:y+p+1, x-p:x+p+1]
            padded[y, x] = window.any()

    return padded


def mask_bb(mask: np.ndarray):
    indices = np.where(mask)
    y_start, y_end = np.min(indices[0]), np.max(indices[0])
    height = (y_end - y_start) + 1
    x_start, x_end = np.min(indices[1]), np.max(indices[1])
    width = (x_end - x_start) + 1
    return (y_start, x_start, height, width)


if __name__ == "__main__":
    boundary_file = "data/boundaries/sphere.png"
    boundary = cv2.imread(boundary_file, cv2.IMREAD_UNCHANGED)
    scale = 4

    mask = mask_from_boundary(boundary)
    y, x, h, w = mask_bb(mask)

    show_scaled("boundary", boundary, scale)
    show_scaled("bb", boundary[y:y+h, x:x+w], scale)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
