import cv2
import numpy as np

from visualizations import show_scaled, visualize_mask


def mask_from_boundary(boundary: np.ndarray):
    # TODO adapt for boundary == edges
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


def canny_test():
    img = cv2.imread('data/trunk_unpatterned.png', cv2.IMREAD_UNCHANGED)
    assert img is not None, "file could not be read, check with os.path.exists()"
    show_scaled("Original", img, 4)

    # for min in range(0, 1000, 100):
    #     for max in range(min, 1000, 100):
    #         edges = cv2.Canny(img, min, max)
    #         show_scaled(f"Edges {min} - {max}", edges, 4)

    for min, max in [(0, 0), (200, 300), (100, 400), (0, 200), (0, 100), (0, 300), (300, 500), (300, 900)]:
        edges = cv2.Canny(img, min, max)
        show_scaled(f"Edges {min} - {max}", edges, 8)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    boundary_file = "data/shaded_tree_canopy_full_boundary.png"
    boundary = cv2.imread(boundary_file, cv2.IMREAD_UNCHANGED)
    scale = 4

    mask = mask_from_boundary(boundary)
    padded1 = pad_mask(mask, 1)
    padded2 = pad_mask(mask, 2)

    masks = visualize_mask(mask, 0) + visualize_mask(padded1,
                                                     1) + visualize_mask(padded2, 2)
    show_scaled("overlaid", masks, scale)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
