import numpy as np

def window_function(img, X, Y, func, window):
    pass


def interpolate_bicubic(img, X, Y):
    f_interp = RectBivariateSpline(
        np.arange(dat.shape[0], dtype=np.float32),
        np.arange(dat.shape[1], dtype=np.float32),
        dat,
    )
    return f_interp(Y, X, grid=False)


def interpolate_Lanczos_grid(img, X, Y, scale):
    """
    Perform Lanczos interpolation at a grid of points.
    https://pixinsight.com/doc/docs/InterpolationAlgorithms/InterpolationAlgorithms.html
    """
    
    sinc_X = list(
        np.sinc(np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i]))
        * np.sinc(
            (np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i])) / scale
        )
        for i in range(len(X))
    )
    sinc_Y = list(
        np.sinc(np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i]))
        * np.sinc(
            (np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i])) / scale
        )
        for i in range(len(Y))
    )

    # Extract an image which has the required dimensions
    use_img = np.take(
        np.take(img, np.arange(int(np.floor(Y[0]) - step + 1), int(np.floor(Y[-1]) + step + 1)), 0, mode = "clip"),
        np.arange(int(np.floor(X[0]) - step + 1), int(np.floor(X[-1]) + step + 1)), 1, mode = "clip"
    )

    # Create a sliding window view of the image with the dimensions of the lanczos scale grid
    #window = np.lib.stride_tricks.sliding_window_view(use_img, (2*scale, 2*scale))

    # fixme going to need some broadcasting magic
    XX = np.ones((2*scale,2*scale))
    res = np.zeros((len(Y), len(X)))
    for x, lowx, highx in zip(range(len(X)), np.floor(X) - step + 1, np.floor(X) + step + 1):
        for y, lowy, highy in zip(range(len(Y)), np.floor(Y) - step + 1, np.floor(Y) + step + 1):
            L = XX * sinc_X[x] * sinc_Y[y].reshape((sinc_Y[y].size, -1))
            res[y,x] = np.sum(use_img[lowy:highy,lowx:highx] * L) / np.sum(L)
    return res
            
def interpolate_Lanczos(img, X, Y, scale):
    """
    Perform Lanczos interpolation on an image at a series of specified points.
    https://pixinsight.com/doc/docs/InterpolationAlgorithms/InterpolationAlgorithms.html
    """
    flux = []

    for i in range(len(X)):
        box = [
            [
                max(0, int(round(np.floor(X[i]) - scale + 1))),
                min(img.shape[1], int(round(np.floor(X[i]) + scale + 1))),
            ],
            [
                max(0, int(round(np.floor(Y[i]) - scale + 1))),
                min(img.shape[0], int(round(np.floor(Y[i]) + scale + 1))),
            ],
        ]
        chunk = img[box[1][0] : box[1][1], box[0][0] : box[0][1]]
        XX = np.ones(chunk.shape)
        Lx = (
            np.sinc(np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i]))
            * np.sinc(
                (np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i])) / scale
            )
        )[
            box[0][0]
            - int(round(np.floor(X[i]) - scale + 1)) : 2 * scale
            + box[0][1]
            - int(round(np.floor(X[i]) + scale + 1))
        ]
        Ly = (
            np.sinc(np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i]))
            * np.sinc(
                (np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i])) / scale
            )
        )[
            box[1][0]
            - int(round(np.floor(Y[i]) - scale + 1)) : 2 * scale
            + box[1][1]
            - int(round(np.floor(Y[i]) + scale + 1))
        ]
        L = XX * Lx * Ly.reshape((Ly.size, -1))
        w = np.sum(L)
        flux.append(np.sum(chunk * L) / w)
    return np.array(flux)

def nearest_neighbor(img, X, Y):
    return img[np.clip(np.asarray(np.rint(Y),dtype=int), a_min = 0, a_max = img.shape[0]),
               np.clip(np.asarray(np.rint(X),dtype=int), a_min = 0, a_max = img.shape[1])
    ]

