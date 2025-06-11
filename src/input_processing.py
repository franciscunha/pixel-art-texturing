import cv2
import numpy as np


# Elements

def split_oriented_spritesheet(img: cv2.Mat):
    """
    Splits a 3x3 spritesheet into individual sprites based on orientation.

    The spritesheet is assumed to have 9 sprites arranged in a 3x3 grid.
    Each sprite represents a different orientation:
        [0,0]: top-left     [0,1]: top       [0,2]: top-right
        [1,0]: left         [1,1]: center    [1,2]: right
        [2,0]: bottom-left  [2,1]: bottom    [2,2]: bottom-right

    Args:
        img: OpenCV image (H,W,C) format where H = sprite_h*3 and W = sprite_w*3

    Returns:
        sprites: Numpy array of shape (3,3,sprite_h,sprite_w,4) 
                 containing all sprites indexed as sprites[row, col]
    """
    # Get total image dimensions
    img_height, img_width, _ = img.shape

    # Calculate individual sprite dimensions
    sprite_height = int(img_height / 3)
    sprite_width = int(img_width / 3)

    # Initialize output array: [row, col, sprite_height, sprite_width, channels]
    sprites = np.empty((3, 3, sprite_height, sprite_width, 4))

    # Iterate through the 3x3 grid
    for row_idx in range(3):
        for col_idx in range(3):
            # This section has an apparent x, y swap. This is because we need to
            # access the image with OpenCV's [y, x] indexing, but the sprites
            # themselves are laid out in the more intuitive [x, y] indexing.

            # Calculate starting pixel coordinates in the original image
            # Convert grid position to pixel position
            start_x = (row_idx) * sprite_height
            start_y = (col_idx) * sprite_width

            # Extract the sprite and store it in our output array
            sprites[row_idx, col_idx, :, :, :] = img[start_y:start_y+sprite_height,
                                                     start_x:start_x+sprite_width, :]

    return sprites


# Mask

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


def cut_to_bb(
    arr: np.ndarray,
    bb: tuple[int, int, int, int] | list[int, int, int, int]
):
    bb_y, bb_x, bb_h, bb_w = bb
    return arr[bb_y:bb_y+bb_h, bb_x:bb_x+bb_w]
