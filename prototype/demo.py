import cv2
import numpy as np

from annotations import get_annotation_coords
from texturing import texture
from vector_field import compress_vector_field
from visualizations import save_scaled, show_scaled, visualize_vector_field

# Params for animal:
# boundary_mask_padding = 2
# pattern_padding = (-3, -3)
# allow_partly_in_mask = False
# placement_mode = "sampling"
# color_map_mode = "border"
# element_color_mode = "per-pixel"


#! Params

source_file = "data/bases/fish/mini_fish.png"
pattern_sheet_file = "data/pattern_sheet/fish_scale.png"
boundary_file = "data/bases/fish/mini_fish.png"

save_output = True
output_name = "mini_fish"

# Vector field
show_annotations = True
show_vector_field = True
grid_scale = (1, 1)
grid_cell_size = 20

# Mask
boundary_mask_padding = 0
show_mask = False

# Placement
placement_mode = "packed"
# placement_mode = "sampling"
allow_partly_in_mask = True
# allow_partly_in_mask = False

# Density
pattern_padding = (-1, -2)
num_patterns = 0

# Coloring
# show_color_map = True
show_color_map = False

# Uncomment appropriate parameters for color mode
hsv_shift = (0, 0, -int(0.3*255))
# color_map_mode = "hsv"
# color_map_mode = "similarity"
color_map_mode = "border"
element_color_mode = "region"
# element_color_mode = "per-pixel"
excluded_colors = np.array([[0, 0, 0, 255], [255, 255, 255, 255]])

# Visualization
scale = 8

#! Loading images

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
    texture(source, pattern_sheet, boundary, num_patterns, placement_mode,
            allow_partly_in_mask, boundary_mask_padding, pattern_padding, scale,
            excluded_colors, color_map_mode, element_color_mode, hsv_shift, result_only=False)


#! Showing and saving results

if show_vector_field:
    annotations_compressed = compress_vector_field(annotations, grid_scale)
    annotated_coords = get_annotation_coords(annotations_compressed)

    vector_field_img = visualize_vector_field(
        compress_vector_field(vector_field, grid_scale),
        input_vector_coords=annotated_coords,
        cell_size=grid_cell_size
    )

    cv2.imshow("Vector field", vector_field_img)
    if save_output:
        cv2.imwrite(
            f"out/{output_name}.vector_field.png", vector_field_img[:, :, :3])
        np.save(f"out/{output_name}.vector_field.npy", vector_field)

if show_annotations:
    annotations_img = visualize_vector_field(
        compress_vector_field(annotations, grid_scale),
        cell_size=grid_cell_size)
    cv2.imshow("Annotations", annotations_img)
    if save_output:
        cv2.imwrite(
            f"out/{output_name}.annotations.png", annotations_img[:, :, :3])
        np.save(f"out/{output_name}.annotations.npy", annotations)

if show_color_map:
    show_scaled("Color map", colors, scale)
    if save_output:
        save_scaled(f"out/{output_name}.color_map.png", colors, scale)

if show_mask:
    mask_img = cv2.bitwise_and(source, source, mask=mask.astype(np.uint8))
    show_scaled("Mask", mask_img, scale)
    if save_output:
        save_scaled(f"out/{output_name}.mask.png", mask_img, scale)

show_scaled("Output", result, scale)
if save_output:
    save_scaled(f"out/{output_name}.result.png", result, scale)


cv2.waitKey(0)
cv2.destroyAllWindows()
