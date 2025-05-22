import numpy as np


def cut_to_bb(arr: np.ndarray, bb: tuple[int, int, int, int] | list[int, int, int, int]):
    bb_y, bb_x, bb_h, bb_w = bb
    return arr[bb_y:bb_y+bb_h, bb_x:bb_x+bb_w]
