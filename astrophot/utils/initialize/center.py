import numpy as np


def center_of_mass(image):
    """Determines the light weighted center of mass"""
    ii, jj = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij")
    center = np.array((np.sum(image * ii), np.sum(image * jj))) / np.sum(image)
    return center


def recursive_center_of_mass(image, max_iter=10, tol=1e-1):
    """Determines the light weighted center of mass in a progressively smaller window each time centered on the previous center."""

    center = center_of_mass(image)
    for i in range(max_iter):
        width = (image.shape[0] / (3 + i), image.shape[1] / (3 + i))
        ranges = (
            slice(
                max(0, int(center[0] - width[0])), min(image.shape[0], int(center[0] + width[0]))
            ),
            slice(
                max(0, int(center[1] - width[1])), min(image.shape[1], int(center[1] + width[1]))
            ),
        )
        subimage = image[ranges]
        if subimage.size < 9:
            return center
        new_center = center_of_mass(subimage)
        new_center += np.array((ranges[0].start, ranges[1].start))

        if np.linalg.norm(new_center - center) < tol:
            return new_center

        center = new_center
    return center
