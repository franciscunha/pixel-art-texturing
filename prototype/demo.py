import cv2
import numpy as np

from annotated_placement import place_patterns_within_boundary, split_oriented_spritesheet
from annotations import compress_vector_field, draw_on_image, parse_curves
from visualizations import visualize_vector_field, show_scaled
from vector_field import diffuse_vector_field


show_annotations = False
show_vector_field = True
grid_scale = (4, 4)

scale = 4

base_file = "data/shaded_tree.png"
pattern_sheet_file = "data/slynrd_leaf_spritesheet.png"
boundary_file = "data/shaded_tree_canopy_full_boundary.png"

base = cv2.imread(base_file, cv2.IMREAD_UNCHANGED)
pattern_sheet = cv2.imread(pattern_sheet_file, cv2.IMREAD_UNCHANGED)
if boundary_file is None:
    boundary = np.full_like(base, 255)
else:
    boundary = cv2.imread(boundary_file, cv2.IMREAD_UNCHANGED)
    if boundary.shape != base.shape:
        raise ValueError("Boundary has to be the same size as base")

if base is None or pattern_sheet is None or boundary is None:
    raise FileNotFoundError()

shape = base.shape[:2]
patterns = split_oriented_spritesheet(pattern_sheet)

curves = draw_on_image(base, scale)
annotations = compress_vector_field(
    parse_curves(curves, shape[0]*scale, shape[1]*scale), (scale, scale)
)

vector_field = diffuse_vector_field(annotations)

if show_vector_field:
    vector_field_img = visualize_vector_field(
        compress_vector_field(vector_field, grid_scale))
    cv2.imshow("Vector field", vector_field_img)

if show_annotations:
    annotations_img = visualize_vector_field(
        compress_vector_field(annotations, grid_scale))
    cv2.imshow("Annotations", annotations_img)

result = place_patterns_within_boundary(base, patterns, boundary, vector_field,
                                        num_patterns=500, hsv_shift=(0, 0, -20))

show_scaled("Output", result, scale)

cv2.waitKey(0)
cv2.destroyAllWindows()
