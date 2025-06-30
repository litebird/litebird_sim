"""
GLS Map-maker using BrahMap for Litebird_sim
This function provides a consistent interface with other mapmaking routines.
"""

from typing import Any, Optional, Union, List, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from litebird_sim.hwp import HWP


if TYPE_CHECKING:  # pragma: no cover
    # Only imported for type checking, not at runtime
    import brahmap


def make_brahmap_gls_map(
    nside: int,
    observations: Union[list, Any],
    pointings: Optional[Union[npt.ArrayLike, List[npt.ArrayLike]]] = None,
    hwp: Optional[HWP] = None,
    components: Union[str, List[str]] = "tod",
    pointings_flag: Optional[np.ndarray] = None,
    inv_noise_cov_operator: Union["brahmap.LBSim_InvNoiseCovLO_UnCorr", None] = None,
    threshold: float = 1.0e-5,
    pointings_dtype=np.float64,
    gls_params: Optional["brahmap.LBSimGLSParameters"] = None,
) -> Union[
    "brahmap.LBSimGLSResult",
    tuple["brahmap.LBSimProcessTimeSamples", "brahmap.LBSimGLSResult"],
]:
    """
    GLS Map-maker using Brahmap.

    This function allows the users to do the optimal map-making with
    BrahMap. Since BrahMap is seemlessly interfaced with LBS, it allows
    the map-making without storing the TODs on the disk. The function
    needs an inverse noise covariance operator and an object containing
    the GLS parameters as arguments.

    BrahMap offers a variety of noise covariance opeartors, all
    compatible with LBS. Noise covariance operator for white noise, for
    example, can be defined as::

        inv_cov = brahmap.LBSim_InvNoiseCovLO_UnCorr(
            obs = ...,
            inverse_noise_variance = ...,
            dtype = ...,
        )

    The parameters used for GLS can be defined as::

        gls_params = brahmap.LBSimGLSParameters(
            output_coordinate_system = lbs.CoordinateSystem.Galactic,
            return_processed_samples = False,
            solver_type = brahmap.SolverType.IQU,
            use_iterative_solver = True,
            isolver_threshold = 1.0e-25,
            isolver_max_iterations = 100,
            return_hit_map = False,
        )

    For the complete list of available noise covariance operators and
    advance usage, please refer to the BrahMap documentation:
    https://anand-avinash.github.io/BrahMap/

    Parameters
    ----------
    nside : int
        Nside of the output map
    observations: Observation
        An instance of :class:`.Observation` class, or a list of objects of
        this kind
    pointings: np.ndarray, optional
        A 3×N array containing the values of the θ,φ,ψ angles (in radians), or
        a list if `observations` was a list. If no pointings are specified,
        they will be taken from `observations` (the most common situation)
    hwp: HWP, optional
        An instance of ``HWP`` class
    components : str or list[str], optional
        The components to be summed for map-making, by default "tod"
    pointings_flag: np.ndarray, optional
        An array of flags to include or exclude the individual pointing
        sample, by default None. A ``False`` flag excludes the pointing sample
        whereas a ``True`` flag includes it
    inv_noise_cov_operator : optional
        Inverse noise covariance operator, by default None
    threshold : float, optional
        Threshold parameter to determine the poorly observed pixels, by
        default 1.0e-5
    pointings_dtype : dtype, optional
        dtype to be used for computing pointings on the fly. The
        `pointing_dtype` must be same as the dtype of the TOD; by default
        `np.float64`
    gls_params : LBSimProcessTimeSamples, optional
        An object that encapsulates the parameter of the GLS, including
        PCG threshold and max iteration; by default None

    Returns
    -------
    Union[LBSimGLSResult, tuple[LBSimProcessTimeSamples, LBSimGLSResult]]
        Returns an `LBSimGLSResult` object when
        `gls_params.return_processed_samples = False`. `LBSimGLSResult`
        object encapsulates the output of GLS including the output maps
        and PCG convergence status. The function returns the tuple object
        `(LBSimProcessTimeSamples, LBSimGLSResult)` when
        `gls_params.return_processed_samples = True`.

    Raises
    ------
    ImportError
        Raises `ImportError` when the brahmap package couldn't be imported
    """
    try:
        import brahmap
    except ImportError:
        raise ImportError(
            "Could not import `BrahMap`. Make sure that the package "
            "`BrahMap` is installed in the same environment "
            "as `litebird_sim`. Refer to "
            "https://anand-avinash.github.io/BrahMap/overview/installation/ "
            "for the installation instruction"
        )

    if gls_params is None:
        gls_params = brahmap.LBSimGLSParameters()

    gls_result = brahmap.LBSim_compute_GLS_maps(
        nside=nside,
        observations=observations,
        pointings=pointings,
        hwp=hwp,
        components=components,
        pointings_flag=pointings_flag,
        inv_noise_cov_operator=inv_noise_cov_operator,
        threshold=threshold,
        dtype_float=pointings_dtype,
        LBSim_gls_parameters=gls_params,
    )
    return gls_result
