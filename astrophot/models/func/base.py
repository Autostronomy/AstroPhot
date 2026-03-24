from ...backend_obj import ArrayLike


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


def downsample(img: ArrayLike, scale: int = 1):
    if scale == 1:
        return img
    MS = img.shape[0] // scale
    NS = img.shape[1] // scale

    return img[: MS * scale, : NS * scale].reshape(MS, scale, NS, scale).sum(axis=(1, 3))


def downsample_mean(img: ArrayLike, scale: int = 1):
    if scale == 1:
        return img
    MS = img.shape[0] // scale
    NS = img.shape[1] // scale

    return img[: MS * scale, : NS * scale].reshape(MS, scale, NS, scale).mean(axis=(1, 3))
