import numpy as np


def fluxdens_to_fluxsum(R, I, axisratio):
    """
    Integrate a flux density profile

    R: semi-major axis length (arcsec)
    I: flux density (flux/arcsec^2)
    axisratio: b/a profile
    """

    S = np.zeros(len(R))
    S[0] = I[0] * np.pi * axisratio[0] * (R[0] ** 2)
    for i in range(1, len(R)):
        S[i] = (
            trapz(2 * np.pi * I[: i + 1] * R[: i + 1] * axisratio[: i + 1], R[: i + 1])
            + S[0]
        )
    return S


def fluxdens_to_fluxsum_errorprop(
    R, I, IE, axisratio, axisratioE=None, N=100, symmetric_error=True
):
    """
    Integrate a flux density profile

    R: semi-major axis length (arcsec)
    I: flux density (flux/arcsec^2)
    axisratio: b/a profile
    """
    if axisratioE is None:
        axisratioE = np.zeros(len(R))

    # Create container for the monte-carlo iterations
    sum_results = np.zeros((N, len(R))) - 99.999
    I_CHOOSE = np.logical_and(np.isfinite(I), I > 0)
    if np.sum(I_CHOOSE) < 5:
        return (None, None) if symmetric_error else (None, None, None)
    sum_results[0][I_CHOOSE] = fluxdens_to_fluxsum(
        R[I_CHOOSE], I[I_CHOOSE], axisratio[I_CHOOSE]
    )
    for i in range(1, N):
        # Randomly sampled SB profile
        tempI = np.random.normal(loc=I, scale=np.abs(IE))
        # Randomly sampled axis ratio profile
        tempq = np.clip(
            np.random.normal(loc=axisratio, scale=np.abs(axisratioE)),
            a_min=1e-3,
            a_max=1 - 1e-3,
        )
        # Compute COG with sampled data
        sum_results[i][I_CHOOSE] = fluxdens_to_fluxsum(
            R[I_CHOOSE], tempI[I_CHOOSE], tempq[I_CHOOSE]
        )

    # Condense monte-carlo evaluations into profile and uncertainty envelope
    sum_lower = sum_results[0] - np.quantile(sum_results, 0.317310507863 / 2, axis=0)
    sum_upper = (
        np.quantile(sum_results, 1.0 - 0.317310507863 / 2, axis=0) - sum_results[0]
    )

    # Return requested uncertainty format
    if symmetric_error:
        return sum_results[0], np.abs(sum_lower + sum_upper) / 2
    else:
        return sum_results[0], sum_lower, sum_upper


def _Fmode_integrand(t, parameters):
    fsum = sum(
        parameters["Am"][m] * np.cos(parameters["m"][m] * (t + parameters["Phim"][m]))
        for m in range(len(parameters["m"]))
    )
    dfsum = sum(
        parameters["m"][m]
        * parameters["Am"][m]
        * np.sin(parameters["m"][m] * (t + parameters["Phim"][m]))
        for m in range(len(parameters["m"]))
    )
    return (np.sin(t) ** 2) * np.exp(2 * fsum) + np.sin(t) * np.cos(t) * np.exp(
        fsum
    ) * dfsum


def Fmode_Areas(R, parameters):
    A = []
    for i in range(len(R)):
        A.append(
            (R[i] ** 2) * quad(_Fmode_integrand, 0, 2 * np.pi, args=(parameters[i],))[0]
        )
    return np.array(A)


def Fmode_fluxdens_to_fluxsum(R, I, parameters, A=None):
    """
    Integrate a flux density profile, with isophotes including Fourier perturbations.

    Arguments
    ---------
    R: arcsec
      semi-major axis length

    I: flux/arcsec^2
      flux density

    parameters: list of dictionaries
      list of dictionary of isophote shape parameters for each radius.
      formatted as

      .. code-block:: python

        {'ellip': ellipticity,
         'm': list of modes used,
         'Am': list of mode powers,
         'Phim': list of mode phases

      }

      entries for each radius.
    """
    if all(parameters[p]["m"] is None for p in range(len(parameters))):
        return fluxdens_to_fluxsum(
            R,
            I,
            1.0
            - np.array(list(parameters[p]["ellip"] for p in range(len(parameters)))),
        )

    S = np.zeros(len(R))
    if A is None:
        A = Fmode_Areas(R, parameters)
    # update the Area calculation to be scaled by the ellipticity
    Aq = A * np.array(list((1 - parameters[i]["ellip"]) for i in range(len(R))))
    S[0] = I[0] * Aq[0]
    Adiff = np.array([Aq[0]] + list(Aq[1:] - Aq[:-1]))
    for i in range(1, len(R)):
        S[i] = trapz(I[: i + 1] * Adiff[: i + 1], R[: i + 1]) + S[0]
    return S


def Fmode_fluxdens_to_fluxsum_errorprop(
    R, I, IE, parameters, N=100, symmetric_error=True
):
    """
    Integrate a flux density profile, with isophotes including Fourier perturbations.

    Arguments
    ---------
    R: arcsec
      semi-major axis length

    I: flux/arcsec^2
      flux density

    parameters: list of dictionaries
      list of dictionary of isophote shape parameters for each radius.
      formatted as

      .. code-block:: python

        {'ellip': ellipticity,
         'm': list of modes used,
         'Am': list of mode powers,
         'Phim': list of mode phases

      }

      entries for each radius.
    """

    for i in range(len(R)):
        if not "ellip err" in parameters[i]:
            parameters[i]["ellip err"] = np.zeros(len(R))
    if all(parameters[p]["m"] is None for p in range(len(parameters))):
        return fluxdens_to_fluxsum_errorprop(
            R,
            I,
            IE,
            1.0
            - np.array(list(parameters[p]["ellip"] for p in range(len(parameters)))),
            np.array(list(parameters[p]["ellip err"] for p in range(len(parameters)))),
            N=N,
            symmetric_error=symmetric_error,
        )

    # Create container for the monte-carlo iterations
    sum_results = np.zeros((N, len(R))) - 99.999
    I_CHOOSE = np.logical_and(np.isfinite(I), I > 0)
    if np.sum(I_CHOOSE) < 5:
        return (None, None) if symmetric_error else (None, None, None)
    cut_parameters = list(compress(parameters, I_CHOOSE))
    A = Fmode_Areas(R[I_CHOOSE], cut_parameters)
    sum_results[0][I_CHOOSE] = Fmode_fluxdens_to_fluxsum(
        R[I_CHOOSE], I[I_CHOOSE], cut_parameters, A
    )
    for i in range(1, N):
        # Randomly sampled SB profile
        tempI = np.random.normal(loc=I, scale=np.abs(IE))
        # Randomly sampled axis ratio profile
        temp_parameters = deepcopy(cut_parameters)
        for p in range(len(cut_parameters)):
            temp_parameters[p]["ellip"] = np.clip(
                np.random.normal(
                    loc=cut_parameters[p]["ellip"],
                    scale=np.abs(cut_parameters[p]["ellip err"]),
                ),
                a_min=1e-3,
                a_max=1 - 1e-3,
            )
        # Compute COG with sampled data
        sum_results[i][I_CHOOSE] = Fmode_fluxdens_to_fluxsum(
            R[I_CHOOSE], tempI[I_CHOOSE], temp_parameters, A
        )

    # Condense monte-carlo evaluations into profile and uncertainty envelope
    sum_lower = sum_results[0] - np.quantile(sum_results, 0.317310507863 / 2, axis=0)
    sum_upper = (
        np.quantile(sum_results, 1.0 - 0.317310507863 / 2, axis=0) - sum_results[0]
    )

    # Return requested uncertainty format
    if symmetric_error:
        return sum_results[0], np.abs(sum_lower + sum_upper) / 2
    else:
        return sum_results[0], sum_lower, sum_upper
