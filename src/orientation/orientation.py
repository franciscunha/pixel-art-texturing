from src.orientation.diffusion import diffuse_vector_field
from src.orientation.vectors import compress_vector_field
from src.orientation.user_annotation import draw_on_image, parse_curves


def orientations(img, scale):
    curves = draw_on_image(img, scale)

    unscaled_annotations = parse_curves(
        curves, img.shape[0]*scale, img.shape[1]*scale)

    annotations = compress_vector_field(unscaled_annotations, (scale, scale))
    vector_field = diffuse_vector_field(annotations)

    return vector_field, annotations
