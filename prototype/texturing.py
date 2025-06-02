
import cv2
import numpy as np
from placement import find_pattern, place_pattern, pattern_positions, split_oriented_spritesheet
from annotations import draw_on_image, parse_curves
from boundaries import mask_bb, mask_from_boundary, pad_mask
from coloring import color_map
from diffusion import diffuse_vector_field
from vector_field import compress_vector_field


def texture(
    source: cv2.Mat,
    pattern_sheet: np.ndarray,
    boundary: cv2.Mat,
    num_patterns: int,
    placement_mode: str = "sampling",
    allow_partly_in_mask: bool = False,
    boundary_mask_padding: int = 0,
    pattern_padding: tuple[int, int] = (0, 0),  # x, y
    annotation_img_scale: int = 1,
    excluded_colors: np.ndarray = [],
    color_map_mode: str = "border",
    element_color_mode: str = "region",
    hsv_shift: tuple[int, int, int] | None = None,
    result_only: bool = True,
):
    if boundary.shape != source.shape:
        raise ValueError("Boundary has to be the same size as source")

    #! Process input

    patterns = split_oriented_spritesheet(pattern_sheet)
    mask = pad_mask(mask_from_boundary(boundary), boundary_mask_padding)

    # TODO handle everything only inside bb
    bb = mask_bb(mask)

    #! Vector field

    scale = annotation_img_scale
    curves = draw_on_image(source, scale)
    annotations = compress_vector_field(
        parse_curves(curves, source.shape[0]*scale,
                     source.shape[1]*scale), (scale, scale)
    )

    vector_field = diffuse_vector_field(annotations)

    #! Coloring

    # TODO adapt HSV shift to also create a color mask
    # if hsv_shift is None:
    colors = color_map(
        source, mask, exclude=excluded_colors, type=color_map_mode)
    # else:
    #     colors = None

    #! Positions

    pattern_shape = patterns.shape[2:4]
    positions = pattern_positions(
        mask, pattern_shape, placement_mode,
        allow_partly_in_mask, pattern_padding, num_patterns)

    #! Placement

    result = source.copy()
    for position in positions:
        pattern = find_pattern(vector_field, position, patterns)
        place_pattern(result, pattern, position, mask,
                      colors, element_color_mode)

    #! Return

    if result_only:
        return result

    return result, mask, colors, annotations, vector_field, positions
