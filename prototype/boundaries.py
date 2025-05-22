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
