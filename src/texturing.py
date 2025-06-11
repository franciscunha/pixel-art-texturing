import cv2
import numpy as np

from orientation import orientations
from color import color_map
from placement import place_elements
from position import pattern_positions
from input_processing import mask_bb, mask_from_boundary, pad_mask, split_oriented_spritesheet


def texture(
    source: cv2.Mat,
    pattern_sheet: np.ndarray,
    boundary: cv2.Mat,
    density: float = 1.0,
    placement_mode: str = "sampling",
    allow_partly_in_mask: bool = False,
    boundary_mask_padding: int = 0,
    pattern_padding: tuple[int, int] = (0, 0),  # x, y
    annotation_img_scale: int = 1,
    excluded_colors: np.ndarray = [],
    color_map_mode: str = "border",
    element_color_mode: str = "region",
    hsv_shift: tuple[int, int, int] | None = None,
    max_attempts: int = 1000,
    result_only: bool = True,
):
    # TODO change terminology to reflect paper

    if boundary.shape != source.shape:
        raise ValueError("Boundary has to be the same size as source")

    #! Process input

    patterns = split_oriented_spritesheet(pattern_sheet)
    mask = pad_mask(mask_from_boundary(boundary), boundary_mask_padding)

    # TODO handle everything only inside bb
    bb = mask_bb(mask)

    #! Vector field

    vector_field, annotations = orientations(source, annotation_img_scale)

    #! Coloring

    if color_map_mode == "hsv" and hsv_shift is None:
        raise ValueError("Specify an HSV shift value for HSV color map mode")

    colors = color_map(
        source,
        mask,
        exclude=excluded_colors,
        type=color_map_mode,
        hsv_shift=hsv_shift
    )

    #! Positions

    pattern_shape = patterns.shape[2:4]

    positions = pattern_positions(
        mask,
        pattern_shape,
        placement_mode,
        allow_partly_in_mask,
        pattern_padding,
        density,
        max_attempts
    )

    #! Placement

    result = place_elements(
        source,
        patterns,
        mask,
        positions,
        colors,
        vector_field,
        element_color_mode
    )

    #! Return

    if result_only:
        return result

    return result, mask, colors, annotations, vector_field, positions
