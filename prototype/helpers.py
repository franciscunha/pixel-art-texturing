import cv2
import numpy as np


def cut_to_bb(arr: np.ndarray, bb: tuple[int, int, int, int] | list[int, int, int, int]):
    bb_y, bb_x, bb_h, bb_w = bb
    return arr[bb_y:bb_y+bb_h, bb_x:bb_x+bb_w]


def flood_fill_mask(image: cv2.Mat, start_point: tuple[int, int]):
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    _, _, mask_padded, _ = cv2.floodFill(
        bgr, None, start_point, (255, 255, 255))
    mask = mask_padded[1:-1, 1:-1]
    return mask
