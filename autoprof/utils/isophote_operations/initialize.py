

def isophote_initialize(image, n_isophotes = 3):

    circ_ellipse_radii = [1.0]
    allphase = []
    mask = results["mask"] if "mask" in results else None
    if not np.any(mask):
        mask = None

    while circ_ellipse_radii[-1] < (len(image) / 2):
        circ_ellipse_radii.append(circ_ellipse_radii[-1] * (1 + 0.2))
        isovals = _iso_extract(
            image,
            circ_ellipse_radii[-1],
            {"ellip": 0.0, "pa": 0.0},
            results["center"],
            more=True,
            mask=mask,
            sigmaclip=True,
            sclip_nsigma=3,
            interp_mask=True,
        )
        coefs = fft(isovals[0])
        allphase.append(coefs[2])
        # Stop when at 3 time background noise
        if (
            np.quantile(isovals[0], 0.8)
            < (
                (options["ap_fit_limit"] + 1 if "ap_fit_limit" in options else 3)
                * results["background noise"]
            )
            and len(circ_ellipse_radii) > 4
        ):
            break
    logging.info(
        "%s: init scale: %f pix" % (options["ap_name"], circ_ellipse_radii[-1])
    )
    # Find global position angle.
    phase = (-Angle_Median(np.angle(allphase[-5:])) / 2) % np.pi
    if "ap_isoinit_pa_set" in options:
        phase = PA_shift_convention(options["ap_isoinit_pa_set"] * np.pi / 180)

    # Find global ellipticity
    test_ellip = np.linspace(0.05, 0.95, 15)
    test_f2 = []
    for e in test_ellip:
        test_f2.append(
            sum(
                list(
                    _fitEllip_loss(
                        e,
                        image,
                        circ_ellipse_radii[-2] * m,
                        phase,
                        results["center"],
                        results["background noise"],
                        mask,
                    )
                    for m in np.linspace(0.8, 1.2, 5)
                )
            )
        )
    ellip = test_ellip[np.argmin(test_f2)]
    res = minimize(
        lambda e, d, r, p, c, n, msk: sum(
            list(
                _fitEllip_loss(_x_to_eps(e[0]), d, r * m, p, c, n, msk)
                for m in np.linspace(0.8, 1.2, 5)
            )
        ),
        x0=_inv_x_to_eps(ellip),
        args=(
            image,
            circ_ellipse_radii[-2],
            phase,
            results["center"],
            results["background noise"],
            mask,
        ),
        method="Nelder-Mead",
        options={
            "initial_simplex": [
                [_inv_x_to_eps(ellip) - 1 / 15],
                [_inv_x_to_eps(ellip) + 1 / 15],
            ]
        },
    )
    if res.success:
        logging.debug(
            "%s: using optimal ellipticity %.3f over grid ellipticity %.3f"
            % (options["ap_name"], _x_to_eps(res.x[0]), ellip)
        )
        ellip = _x_to_eps(res.x[0])
    if "ap_isoinit_ellip_set" in options:
        ellip = options["ap_isoinit_ellip_set"]
    
