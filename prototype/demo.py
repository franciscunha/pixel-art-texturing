import cv2
import numpy as np

from annotations import get_annotation_coords
from texturing import texture
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
show_color_map = False

# Uncomment appropriate parameters for color mode
# hsv_shift = (0, 0, -20)
hsv_shift = None
# color_mode = "similarity"
color_mode = "border"
excluded_colors = np.array([[0, 0, 0, 255]])

# Visualization
scale = 12

#! Loading images

source_file = "data/bases/green_sphere.png"
pattern_sheet_file = "data/pattern_sheet/slynrd_leaf.png"
boundary_file = "data/boundaries/sphere.png"

source = cv2.imread(source_file, cv2.IMREAD_UNCHANGED)
pattern_sheet = cv2.imread(pattern_sheet_file, cv2.IMREAD_UNCHANGED)

if boundary_file is None:
    boundary = np.full_like(source, 255)
else:
    boundary = cv2.imread(boundary_file, cv2.IMREAD_UNCHANGED)

if source is None or pattern_sheet is None or boundary is None:
    raise FileNotFoundError()


#! Call the algorithm

result, mask, colors, annotations, vector_field, positions =\
    texture(source, pattern_sheet, boundary, num_patterns, boundary_mask_padding,
            pattern_padding, scale, excluded_colors, color_mode, hsv_shift, result_only=False)


#! Showing results

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


if show_color_map:
    show_scaled("Color map", colors, scale)


show_scaled("Output", result, scale)

cv2.waitKey(0)
cv2.destroyAllWindows()
