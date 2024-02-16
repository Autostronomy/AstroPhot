from typing import Optional, Union, Dict, Tuple, Any
import io
from copy import deepcopy

import numpy as np
import torch
from torch.autograd.functional import jacobian as torchjac

from ..param import Parameter_Node, Param_Mask
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.interpolate import (
    _shift_Lanczos_kernel_torch,
    simpsons_kernel,
    curvature_kernel,
    interp2d,
)
from ..image import Model_Image, Target_Image, Window, Jacobian_Image, Window_List, PSF_Image
from ..utils.operations import (
    fft_convolve_torch,
    grid_integrate,
    single_quad_integrate,
)
from ..errors import SpecificationConflict
from .core_model import AstroPhot_Model
from .. import AP_config


@default_internal
def angular_metric(self, X, Y, image=None, parameters=None):
    return torch.atan2(Y, X)


@default_internal
def radius_metric(self, X, Y, image=None, parameters=None):
    return torch.sqrt(X**2 + Y**2 + self.softening**2)


@classmethod
def build_parameter_specs(cls, user_specs=None):
    parameter_specs = {}
    for base in cls.__bases__:
        try:
            parameter_specs.update(base.build_parameter_specs())
        except AttributeError:
            pass
    parameter_specs.update(cls.parameter_specs)
    parameter_specs = deepcopy(parameter_specs)
    if isinstance(user_specs, dict):
        for p in user_specs:
            # If the user supplied a parameter object subclass, simply use that as is
            if isinstance(user_specs[p], Parameter_Node):
                parameter_specs[p] = user_specs[p]
            elif isinstance(
                user_specs[p], dict
            ):  # if the user supplied parameter specifications, update the defaults
                parameter_specs[p].update(user_specs[p])
            else:
                parameter_specs[p]["value"] = user_specs[p]

    return parameter_specs


def build_parameters(self):
    for p in self.__class__._parameter_order:
        # skip if the parameter already exists
        if p in self.parameters:
            continue
        # If a parameter object is provided, simply use as-is
        if isinstance(self.parameter_specs[p], Parameter_Node):
            self.parameters.link(self.parameter_specs[p].to())
        elif isinstance(self.parameter_specs[p], dict):
            self.parameters.link(Parameter_Node(p, **self.parameter_specs[p]))
        else:
            raise ValueError(f"unrecognized parameter specification for {p}")


def _sample_init(self, image, parameters, center):
    if self.sampling_mode == "midpoint":
        Coords = image.get_coordinate_meshgrid()
        X, Y = Coords - center[..., None, None]
        mid = self.evaluate_model(X=X, Y=Y, image=image, parameters=parameters)
        kernel = curvature_kernel(AP_config.ap_dtype, AP_config.ap_device)
        # convolve curvature kernel to numericall compute second derivative
        curvature = torch.nn.functional.pad(
            torch.nn.functional.conv2d(
                mid.view(1, 1, *mid.shape),
                kernel.view(1, 1, *kernel.shape),
                padding="valid",
            ),
            (1, 1, 1, 1),
            mode="replicate",
        ).squeeze()
        return mid + curvature, mid
    elif self.sampling_mode == "simpsons":
        Coords = image.get_coordinate_simps_meshgrid()
        X, Y = Coords - center[..., None, None]
        dens = self.evaluate_model(X=X, Y=Y, image=image, parameters=parameters)
        kernel = simpsons_kernel(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        # midpoint is just every other sample in the simpsons grid
        mid = dens[1::2, 1::2]
        simps = torch.nn.functional.conv2d(
            dens.view(1, 1, *dens.shape), kernel, stride=2, padding="valid"
        )
        return mid.squeeze(), simps.squeeze()
    elif "quad" in self.sampling_mode:
        quad_level = int(self.sampling_mode[self.sampling_mode.find(":") + 1 :])
        Coords = image.get_coordinate_meshgrid()
        X, Y = Coords - center[..., None, None]
        res, ref = single_quad_integrate(
            X=X,
            Y=Y,
            image_header=image.header,
            eval_brightness=self.evaluate_model,
            eval_parameters=parameters,
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
            quad_level=quad_level,
        )
        return ref, res
    elif self.sampling_mode == "trapezoid":
        Coords = image.get_coordinate_corner_meshgrid()
        X, Y = Coords - center[..., None, None]
        dens = self.evaluate_model(X=X, Y=Y, image=image, parameters=parameters)
        kernel = (
            torch.ones((1, 1, 2, 2), dtype=AP_config.ap_dtype, device=AP_config.ap_device) / 4.0
        )
        trapz = torch.nn.functional.conv2d(dens.view(1, 1, *dens.shape), kernel, padding="valid")
        trapz = trapz.squeeze()
        kernel = curvature_kernel(AP_config.ap_dtype, AP_config.ap_device)
        curvature = torch.nn.functional.pad(
            torch.nn.functional.conv2d(
                trapz.view(1, 1, *trapz.shape),
                kernel.view(1, 1, *kernel.shape),
                padding="valid",
            ),
            (1, 1, 1, 1),
            mode="replicate",
        ).squeeze()
        return trapz + curvature, trapz

    raise SpecificationConflict(
        f"{self.name} has unknown sampling mode: {self.sampling_mode}. Should be one of: midpoint, simpsons, quad:level, trapezoid"
    )


def _integrate_reference(self, image_data, image_header, parameters):
    return torch.sum(image_data) / image_data.numel()


def _sample_integrate(self, deep, reference, image, parameters, center):
    if self.integrate_mode == "none":
        pass
    elif self.integrate_mode == "threshold":
        Coords = image.get_coordinate_meshgrid()
        X, Y = Coords - center[..., None, None]
        ref = self._integrate_reference(
            deep, image.header, parameters
        )  # fixme, error can be over 100% on initial sampling reference is invalid
        error = torch.abs((deep - reference))
        select = error > (self.sampling_tolerance * ref)
        intdeep = grid_integrate(
            X=X[select],
            Y=Y[select],
            image_header=image.header,
            eval_brightness=self.evaluate_model,
            eval_parameters=parameters,
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
            quad_level=self.integrate_quad_level,
            gridding=self.integrate_gridding,
            max_depth=self.integrate_max_depth,
            reference=self.sampling_tolerance * ref,
        )
        deep[select] = intdeep
    else:
        raise SpecificationConflict(
            f"{self.name} has unknown integration mode: {self.integrate_mode}. Should be one of: none, threshold"
        )
    return deep


def _shift_psf(self, psf, shift, shift_method="bilinear", keep_pad=True):
    if shift_method == "bilinear":
        psf_data = torch.nn.functional.pad(psf.data, (1, 1, 1, 1))
        X, Y = torch.meshgrid(
            torch.arange(
                psf_data.shape[1],
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            - shift[0],
            torch.arange(
                psf_data.shape[0],
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            - shift[1],
            indexing="xy",
        )
        shift_psf = interp2d(psf_data, X.clone(), Y.clone())
        if not keep_pad:
            shift_psf = shift_psf[1:-1, 1:-1]

    elif "lanczos" in shift_method:
        lanczos_order = int(shift_method[shift_method.find(":") + 1 :])
        psf_data = torch.nn.functional.pad(
            psf.data, (lanczos_order, lanczos_order, lanczos_order, lanczos_order)
        )
        LL = _shift_Lanczos_kernel_torch(
            -shift[0],
            -shift[1],
            lanczos_order,
            AP_config.ap_dtype,
            AP_config.ap_device,
        )
        shift_psf = torch.nn.functional.conv2d(
            psf_data.view(1, 1, *psf_data.shape),
            LL.view(1, 1, *LL.shape),
            padding="same",
        ).squeeze()
        if not keep_pad:
            shift_psf = shift_psf[lanczos_order:-lanczos_order, lanczos_order:-lanczos_order]
    else:
        raise SpecificationConflict(f"unrecognized subpixel shift method: {shift_method}")
    return shift_psf


def _sample_convolve(self, image, shift, psf, shift_method="bilinear"):
    """
    image: Image object with image.data pixel matrix
    shift: the amount of shifting to do in pixel units
    psf: a PSF_Image object
    """
    if shift is not None:
        shift_psf = self._shift_psf(psf, shift, shift_method)
    else:
        shift_psf = psf.data
    shift_psf = shift_psf / torch.sum(shift_psf)

    if self.psf_convolve_mode == "fft":
        image.data = fft_convolve_torch(image.data, shift_psf, img_prepadded=True)
    elif self.psf_convolve_mode == "direct":
        image.data = torch.nn.functional.conv2d(
            image.data.view(1, 1, *image.data.shape),
            torch.flip(
                shift_psf.view(1, 1, *shift_psf.shape),
                dims=(2, 3),
            ),
            padding="same",
        ).squeeze()
    else:
        raise ValueError(f"unrecognized psf_convolve_mode: {self.psf_convolve_mode}")


@torch.no_grad()
def jacobian(
    self,
    parameters: Optional[torch.Tensor] = None,
    as_representation: bool = False,
    window: Optional[Window] = None,
    pass_jacobian: Optional[Jacobian_Image] = None,
    **kwargs,
):
    """Compute the Jacobian matrix for this model.

    The Jacobian matrix represents the partial derivatives of the
    model's output with respect to its input parameters. It is useful
    in optimization and model fitting processes. This method
    simplifies the process of computing the Jacobian matrix for
    astronomical image models and is primarily used by the
    Levenberg-Marquardt algorithm for model fitting tasks.

    Args:
      parameters (Optional[torch.Tensor]): A 1D parameter tensor to override the
                                           current model's parameters.
      as_representation (bool): Indicates if the parameters argument is
                                provided as real values or representations
                                in the (-inf, inf) range. Default is False.
      parameters_identity (Optional[tuple]): Specifies which parameters are to be
                                             considered in the computation.
      window (Optional[Window]): A window object specifying the region of interest
                                 in the image.
      **kwargs: Additional keyword arguments.

    Returns:
      Jacobian_Image: A Jacobian_Image object containing the computed Jacobian matrix.

    """
    if window is None:
        window = self.window
    else:
        if isinstance(window, Window_List):
            window = window.window_list[pass_jacobian.index(self.target)]
        window = self.window & window

    # skip jacobian calculation if no parameters match criteria
    if torch.sum(self.parameters.vector_mask()) == 0 or window.overlap_frac(self.window) <= 0:
        return self.target[window].jacobian_image()

    # Set the parameters if provided and check the size of the parameter list
    if parameters is not None:
        if as_representation:
            self.parameters.vector_set_representation(parameters)
        else:
            self.parameters.vector_set_values(parameters)
    if torch.sum(self.parameters.vector_mask()) > self.jacobian_chunksize:
        return self._chunk_jacobian(
            as_representation=as_representation,
            window=window,
            **kwargs,
        )
    if torch.max(window.pixel_shape) > self.image_chunksize:
        return self._chunk_image_jacobian(
            as_representation=as_representation,
            window=window,
            **kwargs,
        )

    # Compute the jacobian
    full_jac = torchjac(
        lambda P: self(
            image=None,
            parameters=P,
            as_representation=as_representation,
            window=window,
        ).data,
        self.parameters.vector_representation().detach()
        if as_representation
        else self.parameters.vector_values().detach(),
        strategy="forward-mode",
        vectorize=True,
        create_graph=False,
    )

    # Store the jacobian as a Jacobian_Image object
    jac_img = self.target[window].jacobian_image(
        parameters=self.parameters.vector_identities(),
        data=full_jac,
    )
    return jac_img


@torch.no_grad()
def _chunk_image_jacobian(
    self,
    as_representation: bool = False,
    parameters_identity: Optional[tuple] = None,
    window: Optional[Window] = None,
    **kwargs,
):
    """Evaluates the Jacobian in smaller chunks to reduce memory usage.

    For models acting on large windows it can be prohibitive to build
    the full Jacobian in a single pass. Instead this function breaks
    the image into chunks as determined by `self.image_chunksize`
    evaluates the Jacobian only for the sub-images, it then builds up
    the full Jacobian as a separate tensor.

    This is for internal use and should be called by the
    `self.jacobian` function when appropriate.

    """

    pids = self.parameters.vector_identities()
    jac_img = self.target[window].jacobian_image(
        parameters=pids,
    )

    pixel_shape = window.pixel_shape.detach().cpu().numpy()
    Ncells = np.int64(np.round(np.ceil(pixel_shape / self.image_chunksize)))
    cellsize = np.int64(np.round(window.pixel_shape / Ncells))

    for nx in range(Ncells[0]):
        for ny in range(Ncells[1]):
            subwindow = window.copy()
            subwindow.crop_to_pixel(
                (
                    (cellsize[0] * nx, min(pixel_shape[0], cellsize[0] * (nx + 1))),
                    (cellsize[1] * ny, min(pixel_shape[1], cellsize[1] * (ny + 1))),
                )
            )
            jac_img += self.jacobian(
                parameters=None,
                as_representation=as_representation,
                window=subwindow,
                **kwargs,
            )

    return jac_img


@torch.no_grad()
def _chunk_jacobian(
    self,
    as_representation: bool = False,
    parameters_identity: Optional[tuple] = None,
    window: Optional[Window] = None,
    **kwargs,
):
    """Evaluates the Jacobian in small chunks to reduce memory usage.

    For models with many parameters it can be prohibitive to build the
    full Jacobian in a single pass. Instead this function breaks the
    list of parameters into chunks as determined by
    `self.jacobian_chunksize` evaluates the Jacobian only for those,
    it then builds up the full Jacobian as a separate tensor. This is
    for internal use and should be called by the `self.jacobian`
    function when appropriate.

    """
    pids = self.parameters.vector_identities()
    jac_img = self.target[window].jacobian_image(
        parameters=pids,
    )

    for ichunk in range(0, len(pids), self.jacobian_chunksize):
        mask = torch.zeros(len(pids), dtype=torch.bool, device=AP_config.ap_device)
        mask[ichunk : ichunk + self.jacobian_chunksize] = True
        with Param_Mask(self.parameters, mask):
            jac_img += self.jacobian(
                parameters=None,
                as_representation=as_representation,
                window=window,
                **kwargs,
            )

    return jac_img


def load(self, filename: Union[str, dict, io.TextIOBase] = "AstroPhot.yaml", new_name=None):
    """Used to load the model from a saved state.

    Sets the model window to the saved value and updates all
    parameters with the saved information. This overrides the
    current parameter settings.

    Args:
        filename: The source from which to load the model parameters. Can be a string (the name of the file on disc), a dictionary (formatted as if from self.get_state), or an io.TextIOBase (a file stream to load the file from).

    """
    state = AstroPhot_Model.load(filename)
    if new_name is None:
        new_name = state["name"]
    self.name = new_name
    # Use window saved state to initialize model window
    self.window = Window(**state["window"])
    # reassign target in case a target list was given
    self._target_identity = state["target_identity"]
    self.target = self.target
    # Set any attributes which were not default
    for key in self.track_attrs:
        if key in state:
            setattr(self, key, state[key])
    # Load the parameter group, this is handled by the parameter group object
    if isinstance(state["parameters"], Parameter_Node):
        self.parameters = state["parameters"]
    else:
        self.parameters = Parameter_Node(self.name, state=state["parameters"])
    # Move parameters to the appropriate device and dtype
    self.parameters.to(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
    # Re-create the aux PSF model if there was one
    if "psf" in state:
        if state["psf"].get("type", "AstroPhot_Model") == "PSF_Image":
            self.psf = PSF_Image(state=state["psf"])
        else:
            print(state["psf"])
            state["psf"]["parameters"] = self.parameters[state["psf"]["name"]]
            self.set_aux_psf(
                AstroPhot_Model(
                    name=state["psf"]["name"],
                    filename=state["psf"],
                    target=self.target,
                )
            )
    return state
