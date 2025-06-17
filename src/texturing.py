import cv2
import numpy as np

from src.color import color_map
from src.placement import place_elements
from src.position import element_positions
from src.input_processing import process_input, read_files
from src.orientation.orientation import orientations


def annotate(
    source_file: str,
    element_sheet_file: str,
    boundary_file: str,
    boundary_mask_padding=0,
    annotation_scale=1
):
    # Process input

    source, element_sheet, boundary = read_files(
        source_file, element_sheet_file, boundary_file)

    elements, mask, bb = process_input(
        element_sheet, boundary, source.shape, boundary_mask_padding)

    # Annotate

    vector_field, annotations = orientations(source, annotation_scale)

    return source, elements, mask, bb, vector_field, annotations


def texture(
    source: cv2.Mat,
    mask: np.ndarray,
    elements: np.ndarray,
    vector_field: np.ndarray,
    density: float = 1.0,
    placement_mode: str = "sampling",
    allow_partly_in_mask: bool = False,
    element_padding: tuple[int, int] = (0, 0),  # x, y
    excluded_colors: np.ndarray = [],
    color_map_mode: str = "auto",
    element_color_mode: str = "region",
    max_color_distance: float | None = 10,
    hsv_shift: tuple[int, int, int] | None = None,
    max_attempts: int = 1000,
    result_only: bool = True,
):
    #! Coloring

    if (color_map_mode == "hsv" or color_map_mode == "auto") \
            and hsv_shift is None:
        raise ValueError("Specify an HSV shift value for HSV color map mode")

    colors = color_map(
        source,
        mask,
        exclude=excluded_colors,
        type=color_map_mode,
        max_distance=max_color_distance,
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

    # return result, mask, colors, annotations, vector_field, positions
    return result, colors, positions
