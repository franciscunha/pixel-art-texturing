import cv2
import numpy as np
from scipy import sparse

from annotations import compress_vector_field, draw_on_image, parse_curves, visualize_vector_field


def solve_poisson(constraints: sparse.coo_matrix):
    """
    Solve the discrete Poisson equation using SciPy's linear system solver.
    Based on Section 3.2 of Bezerra et al., 2010 (https://doi.org/10.1145/1809939.1809944).
    Note that the inputs and results are all one dimensional, such that you have to e.g.
    call this for each color channel (or vector component).

    Args:
        constraints: Sparse matrix where each filled element constrains 
        the result's value in the same position to be equal to it.
    """
    h, w = constraints.shape
    n = h * w  # Total number of pixels

    constraints = constraints.todok()

    # Initialize off-diagonal elements (-1's)
    i_indices = []
    j_indices = []
    values = []

    # For each pixel, connect to its four neighbors
    for i in range(h):
        for j in range(w):
            idx = i * w + j

            # ifs create Neumann boundary

            # left neighbor
            if j > 0:
                i_indices.append(idx)
                j_indices.append(idx - 1)
                values.append(-1)

            # right neighbor
            if j < w - 1:
                i_indices.append(idx)
                j_indices.append(idx + 1)
                values.append(-1)

            # top neighbor
            if i > 0:
                i_indices.append(idx)
                j_indices.append(idx - w)
                values.append(-1)

            # bottom neighbor
            if i < h - 1:
                i_indices.append(idx)
                j_indices.append(idx + w)
                values.append(-1)

    connections = sparse.coo_matrix(
        (values, (i_indices, j_indices)), shape=(n, n))

    # Initialize diagonal elements (4's)
    diagonal = sparse.eye(n) * 4

    # Create discrete Laplace operator matrix
    laplace = diagonal + connections

    # Initialize divergences (right-hand side) with zeros
    divergences = np.zeros(n)

    # Set constraints
    nonzero_is, nonzero_js = constraints.nonzero()
    for i, j in list(zip(nonzero_is, nonzero_js)):
        idx = i * w + j

        # Reset row in Laplace matrix to influence only the pixel instead of also its neighbors
        laplace[idx, :] = 0  # Clear the row
        laplace[idx, idx] = 1  # Set diagonal to 1

        # Make the pixel equal the constrained value
        divergences[idx] = constraints[i, j]

    # Solve the linear system
    result = sparse.linalg.spsolve(laplace, divergences)

    # Reshape result to image dimensions
    return result.reshape(h, w)


if __name__ == "__main__":
    shape = (32, 32)
    scale = 4

    canvas = np.zeros((shape[0], shape[1], 3), np.uint8)

    curves = draw_on_image(canvas, scale)
    influences = compress_vector_field(
        parse_curves(curves, shape[0]*scale, shape[1]*scale), (scale, scale)
    )

    influences_img = visualize_vector_field(influences)
    cv2.imshow("Influences", influences_img)

    vector_field = np.zeros_like(influences)
    for component in range(2):
        component_wise_influences = sparse.coo_matrix(
            influences[:, :, component])

        vector_field[:, :, component] = \
            solve_poisson(component_wise_influences)

    vector_field_imgs = visualize_vector_field(vector_field)
    cv2.imshow("Vector field", vector_field_imgs)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
