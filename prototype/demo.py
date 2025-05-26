import cv2
import numpy as np

from annotated_placement import place_patterns_within_boundary, split_oriented_spritesheet
from annotations import draw_on_image, get_annotation_coords, parse_curves
from diffusion import diffuse_vector_field
from boundaries import mask_bb, mask_from_boundary, pad_mask
from coloring import color_map
from vector_field import compress_vector_field
from visualizations import show_scaled, visualize_vector_field

#! Params

# Vector field
show_annotations = False
show_vector_field = False
grid_scale = (2, 2)
grid_cell_size = 24

# Mask
boundary_mask_padding = 0

# Density
pattern_padding = -1
num_patterns = 200

# Coloring
show_color_map = True

# Uncomment appropriate parameters for color mode
# hsv_shift = (0, 0, -20)
hsv_shift = None
# color_mode = "similarity"
color_mode = "border"

# Visualization
scale = 6

#! Loading images

base_file = "data/bases/trunk_unpatterned.png"
pattern_sheet_file = "data/pattern_sheet/bark_scale.png"
boundary_file = "data/boundaries/trunk_partial.png"

base = cv2.imread(base_file, cv2.IMREAD_UNCHANGED)
pattern_sheet = cv2.imread(pattern_sheet_file, cv2.IMREAD_UNCHANGED)
if boundary_file is None:
    boundary = np.full_like(base, 255)
else:
    boundary = cv2.imread(boundary_file, cv2.IMREAD_UNCHANGED)

if base is None or pattern_sheet is None or boundary is None:
    raise FileNotFoundError()

if boundary.shape != base.shape:
    raise ValueError("Boundary has to be the same size as base")

shape = base.shape[:2]

#! Process input

patterns = split_oriented_spritesheet(pattern_sheet)
mask = pad_mask(mask_from_boundary(boundary), boundary_mask_padding)

# TODO handle everything only inside bb
bb = mask_bb(mask)

#! Vector field

curves = draw_on_image(base, scale)
annotations = compress_vector_field(
    parse_curves(curves, shape[0]*scale, shape[1]*scale), (scale, scale)
)

vector_field = diffuse_vector_field(annotations)

if show_vector_field:
    annotations_compressed = compress_vector_field(annotations, grid_scale)
    annotated_coords = get_annotation_coords(annotations_compressed)

    vector_field_img = visualize_vector_field(
        compress_vector_field(vector_field, grid_scale),
        input_vector_coords=annotated_coords,
        cell_size=grid_cell_size
    )

    cv2.imshow("Vector field", vector_field_img)

if show_annotations:
    annotations_img = visualize_vector_field(
        compress_vector_field(annotations, grid_scale),
        cell_size=grid_cell_size)
    cv2.imshow("Annotations", annotations_img)

#! Coloring

if hsv_shift is None:
    colors = color_map(base, mask, color_mode)

    if show_color_map:
        show_scaled("Color map", colors, scale)
else:
    colors = None

#! Placement

result = place_patterns_within_boundary(
    base, patterns, mask, vector_field,
    pattern_padding=pattern_padding, num_patterns=num_patterns,
    hsv_shift=hsv_shift, color_map=colors)

#! Output

show_scaled("Output", result, scale)

cv2.waitKey(0)
cv2.destroyAllWindows()
