import numpy as np
from scipy.signal import convolve2d
from autoprof.utils.isophote_operations import _iso_extract

def StarFind(
    IMG,
    fwhm_guess,
    background_noise,
    mask=None,
    peakmax=None,
    detect_threshold=20.0,
    minsep=10.0,
    reject_size=10.0,
    maxstars=np.inf,
):
    """
    Find stars in an image, determine their fwhm and peak flux values.

    IMG: image data as numpy 2D array
    fwhm_guess: A guess at the PSF fwhm, can be within a factor of 2 and everything should work
    background_noise: image background flux noise
    mask: masked pixels as numpy 2D array with same dimensions as IMG
    peakmax: maximum allowed peak flux value for a star, used to remove saturated pixels
    detect_threshold: threshold (in units of sigma) value for convolved image to consider a pixel as a star candidate.
    minsep: minimum allowed separation between stars, in units of fwhm_guess
    reject_size: reject stars with fitted FWHM greater than this times the fwhm_guess
    maxstars: stop once this number of stars have been found, this is for speed purposes
    """

    # Convolve edge detector with image
    S = 3 ** np.array([1, 2, 3, 4, 5])
    S = int(S[np.argmin(np.abs(S / 3 - fwhm_guess))])
    zz = np.ones((S, S)) * -1
    zz[int(S / 3) : int(2 * S / 3), int(S / 3) : int(2 * S / 3)] = 8

    new = convolve2d(IMG, zz, mode="same")

    centers = np.array([])
    deformities = []
    fwhms = []
    peaks = []

    # Select pixels which edge detector identifies
    if mask is None:
        highpixels = np.argwhere(new > detect_threshold * iqr(new))
    else:
        highpixels = np.argwhere(
            np.logical_and(new > detect_threshold * iqr(new), np.logical_not(mask))
        )
    np.random.shuffle(highpixels)
    # meshgrid for 2D polynomial fit (pre-built for efficiency)
    xx, yy = np.meshgrid(np.arange(6), np.arange(6))
    xx = xx.flatten()
    yy = yy.flatten()
    A = np.array(
        [
            np.ones(xx.shape),
            xx,
            yy,
            xx ** 2,
            yy ** 2,
            xx * yy,
            xx * yy ** 2,
            yy * xx ** 2,
            xx ** 2 * yy ** 2,
        ]
    ).T

    for i in range(len(highpixels)):
        # reject if near an existing center
        if len(centers) != 0 and np.any(
            np.sqrt(np.sum((highpixels[i] - centers) ** 2, axis=1))
            < minsep * fwhm_guess
        ):
            continue
        # reject if near edge
        if np.any(highpixels[i] < 5 * fwhm_guess) or np.any(
            highpixels[i] > (np.array(IMG.shape) - 5 * fwhm_guess)
        ):
            continue
        # set starting point at local maximum pixel
        newcenter = np.array([highpixels[i][1], highpixels[i][0]])
        ranges = [
            [
                max(0, int(newcenter[0] - fwhm_guess * 5)),
                min(IMG.shape[1], int(newcenter[0] + fwhm_guess * 5)),
            ],
            [
                max(0, int(newcenter[1] - fwhm_guess * 5)),
                min(IMG.shape[0], int(newcenter[1] + fwhm_guess * 5)),
            ],
        ]
        newcenter = np.unravel_index(
            np.argmax(IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]].T),
            IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]].T.shape,
        )
        newcenter += np.array([ranges[0][0], ranges[1][0]])
        if np.any(newcenter < 5 * fwhm_guess) or np.any(
            newcenter > (np.array(IMG.shape) - 5 * fwhm_guess)
        ):
            continue
        # update star center with 2D polynomial fit
        ranges = [
            [max(0, int(newcenter[0] - 3)), min(IMG.shape[1], int(newcenter[0] + 3))],
            [max(0, int(newcenter[1] - 3)), min(IMG.shape[0], int(newcenter[1] + 3))],
        ]
        chunk = np.clip(
            IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]].T,
            a_min=background_noise / 3,
            a_max=None,
        )
        poly2dfit = np.linalg.lstsq(A, np.log10(chunk.flatten()), rcond=None)
        newcenter = np.array(
            [
                -poly2dfit[0][2] / (2 * poly2dfit[0][4]),
                -poly2dfit[0][1] / (2 * poly2dfit[0][3]),
            ]
        )
        # reject if 2D polynomial maximum is outside the fitting region
        if np.any(newcenter < 0) or np.any(newcenter > 5):
            continue
        newcenter += np.array([ranges[0][0], ranges[1][0]])

        # reject centers that are outside the image
        if np.any(newcenter < 5 * fwhm_guess) or np.any(
            newcenter > (np.array(list(reversed(IMG.shape))) - 5 * fwhm_guess)
        ):
            continue
        # reject stars with too high flux
        if (not peakmax is None) and np.any(
            IMG[
                int(newcenter[1] - minsep * fwhm_guess) : int(
                    newcenter[1] + minsep * fwhm_guess
                ),
                int(newcenter[0] - minsep * fwhm_guess) : int(
                    newcenter[0] + minsep * fwhm_guess
                ),
            ]
            >= peakmax
        ):
            continue
        # reject if near existing center
        if len(centers) != 0 and np.any(
            np.sqrt(np.sum((newcenter - centers) ** 2, axis=1)) < minsep * fwhm_guess
        ):
            continue

        # Extract flux as a function of radius
        local_flux = np.median(
            _iso_extract(
                IMG,
                reject_size * fwhm_guess,
                {"ellip": 0.0, "pa": 0.0},
                {"x": newcenter[0], "y": newcenter[1]},
                interp_method = 'bicubic',
            )
        )
        flux = [
            np.median(
                _iso_extract(
                    IMG,
                    0.0,
                    {"ellip": 0.0, "pa": 0.0},
                    {"x": newcenter[0], "y": newcenter[1]},
                    interp_method = 'bicubic',
                )
            )
            - local_flux
        ]
        if (flux[0] - local_flux) < (detect_threshold * background_noise):
            continue
        R = [0.0]
        deformity = [0.0]
        badcount = 0
        while (flux[-1] > max(flux[0] / 2, background_noise) or len(R) < 5) and len(
            R
        ) < 50:  # len(R) < 50 and (flux[-1] > background_noise or len(R) <= 5):
            R.append(R[-1] + fwhm_guess / 10)
            try:
                isovals = _iso_extract(
                    IMG,
                    R[-1],
                    {"ellip": 0.0, "pa": 0.0},
                    {"x": newcenter[0], "y": newcenter[1]},
                    interp_method = 'bicubic',
                )
            except:
                R = np.zeros(101)  # cause finder to skip this star
                break
            coefs = fft(isovals)
            deformity.append(
                np.sum(np.abs(coefs[1:5]))
                / (len(isovals) * (max(np.median(isovals), 0) + background_noise))
            )  # np.sqrt(np.abs(coefs[0]))
            # if np.sum(np.abs(coefs[1:5])) > np.sqrt(np.abs(coefs[0])):
            #     badcount += 1
            flux.append(np.median(isovals) - local_flux)
        if len(R) >= 50:
            continue
        fwhm_fit = np.interp(flux[0] / 2, list(reversed(flux)), list(reversed(R))) * 2

        # reject if fitted FWHM unrealistically large
        if fwhm_fit > reject_size * fwhm_guess:
            continue
        # Add star to list
        if len(centers) == 0:
            centers = np.array([deepcopy(newcenter)])
        else:
            centers = np.concatenate((centers, [newcenter]), axis=0)
        deformities.append(deformity[-1])
        fwhms.append(deepcopy(fwhm_fit))
        peaks.append(flux[0])
        # stop if max N stars reached
        if len(fwhms) >= maxstars:
            break
        
    return {
        "x": centers[:, 0],
        "y": centers[:, 1],
        "fwhm": np.array(fwhms),
        "peak": np.array(peaks),
        "deformity": np.array(deformities),
    }
