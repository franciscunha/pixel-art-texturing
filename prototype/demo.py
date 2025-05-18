import cv2
import numpy as np

from annotated_placement import place_patterns_within_boundary, split_oriented_spritesheet
from annotations import draw_on_image, parse_curves
from diffusion import diffuse_vector_field
from boundaries import mask_from_boundary, pad_mask
from vector_field import compress_vector_field
from visualizations import show_scaled, visualize_vector_field


show_annotations = False
show_vector_field = True
grid_scale = (4, 4)

boundary_mask_padding = 0
num_patterns = 200
hsv_shift = (0, 0, -20)
# hsv_shift = None

scale = 8

base_file = "data/green_sphere_whitebg.png"
pattern_sheet_file = "data/slynrd_leaf_spritesheet.png"
boundary_file = "data/sphere_boundary.png"

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
patterns = split_oriented_spritesheet(pattern_sheet)

curves = draw_on_image(base, scale)
annotations = compress_vector_field(
    parse_curves(curves, shape[0]*scale, shape[1]*scale), (scale, scale)
)

vector_field = diffuse_vector_field(annotations)

if show_vector_field:
    annotations_compressed = compress_vector_field(annotations, grid_scale)
    annotated_coords = [(y, x)
                        for y, x in np.ndindex(annotations_compressed.shape[:2])
                        if np.any(annotations_compressed[y, x] != 0)]

    vector_field_img = visualize_vector_field(
        compress_vector_field(vector_field, grid_scale),
        input_vector_coords=annotated_coords
    )

    cv2.imshow("Vector field", vector_field_img)

if show_annotations:
    annotations_img = visualize_vector_field(
        compress_vector_field(annotations, grid_scale))
    cv2.imshow("Annotations", annotations_img)

mask = pad_mask(mask_from_boundary(boundary), boundary_mask_padding)

result = place_patterns_within_boundary(
    base, patterns, mask, vector_field, num_patterns=num_patterns, hsv_shift=hsv_shift)

show_scaled("Output", result, scale)

cv2.waitKey(0)
cv2.destroyAllWindows()
