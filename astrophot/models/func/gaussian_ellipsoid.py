from ...backend_obj import backend, ArrayLike


def euler_rotation_matrix(alpha: ArrayLike, beta: ArrayLike, gamma: ArrayLike) -> ArrayLike:
    """Compute the rotation matrix from Euler angles.

    See the Z_alpha X_beta Z_gamma convention for the order of rotations here:
    https://en.wikipedia.org/wiki/Euler_angles
    """
    ca = backend.cos(alpha)
    sa = backend.sin(alpha)
    cb = backend.cos(beta)
    sb = backend.sin(beta)
    cg = backend.cos(gamma)
    sg = backend.sin(gamma)
    R = backend.stack(
        (
            backend.stack((ca * cg - cb * sa * sg, -ca * sg - cb * cg * sa, sb * sa)),
            backend.stack((cg * sa + ca * cb * sg, ca * cb * cg - sa * sg, -ca * sb)),
            backend.stack((sb * cg, sb * cg, cb)),
        ),
        dim=-1,
    )
    return R
