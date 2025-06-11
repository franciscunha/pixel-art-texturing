import cv2
import numpy as np

from src.color import color_map
from src.placement import place_elements
from src.position import element_positions
from src.input_processing import mask_bb, mask_from_boundary, pad_mask, split_oriented_spritesheet
from src.orientation.orientation import orientations


def texture(
    source: cv2.Mat,
    element_sheet: np.ndarray,
    boundary: cv2.Mat,
    density: float = 1.0,
    placement_mode: str = "sampling",
    allow_partly_in_mask: bool = False,
    boundary_mask_padding: int = 0,
    element_padding: tuple[int, int] = (0, 0),  # x, y
    annotation_img_scale: int = 1,
    excluded_colors: np.ndarray = [],
    color_map_mode: str = "border",
    element_color_mode: str = "region",
    hsv_shift: tuple[int, int, int] | None = None,
    max_attempts: int = 1000,
    result_only: bool = True,
):
    if boundary.shape != source.shape:
        raise ValueError("Boundary has to be the same size as source")

    #! Process input

    elements = split_oriented_spritesheet(element_sheet)
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

    element_shape = elements.shape[2:4]

    positions = element_positions(
        mask,
        element_shape,
        placement_mode,
        allow_partly_in_mask,
        element_padding,
        density,
        max_attempts
    )

    #! Placement

    result = place_elements(
        source,
        elements,
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
