# -*- encoding: utf-8 -*-

from enum import IntEnum

from numba import njit, prange
import numpy as np

from typing import Union, List

from .observations import Observation
from .spacecraft import SpacecraftPositionAndVelocity

from litebird_sim import constants as c


# We use a IntEnum class so that comparisons are much faster than with strings
class DipoleType(IntEnum):
    """Approximation for the Doppler shift caused by the motion of the spacecraft"""

    LINEAR = 0
    r"""Linear approximation in β using thermodynamic units:

    .. math:: \Delta T(\vec\beta, \hat n) = T_0 \vec\beta\cdot\hat n

    """ ""

    QUADRATIC_EXACT = 1
    r"""Second-order approximation in β using thermodynamic units:

    .. math:: \Delta T(\vec\beta, \hat n) = T_0\left(\vec\beta\cdot\hat n +
              \bigl(\vec\beta\cdot\hat n\bigr)^2\right)

    """

    TOTAL_EXACT = 2
    r"""Exact formula in true thermodynamic units:

    .. math:: \frac{T_0}{\gamma \bigl(1 - \vec\beta \cdot \hat n\bigr)}
    """

    QUADRATIC_FROM_LIN_T = 3
    r"""Second-order approximation in β using linearized units:

    .. math:: \Delta_2 T(\nu) = T_0 \left(\vec\beta\cdot\hat n + q(x)
              \bigl(\vec\beta\cdot\hat n\bigr)^2\right)
    """

    TOTAL_FROM_LIN_T = 4
    r"""Full formula in linearized units (the most widely used):

    .. math::

       \Delta T = \frac{T_0}{f(x)} \left(\frac{\mathrm{BB}\left(T_0 /
       \gamma\bigl(1 - \vec\beta\cdot\hat n\bigr)\right)}{\mathrm{BB}(T_0)}
       - 1\right) = \frac{T_0}{f(x)} \left(\frac{\mathrm{BB}\bigl(\nu
       \gamma(1-\vec\beta\cdot\hat n), T_0\bigr)}{\bigl(\gamma(1-
       \vec\beta\cdot\hat n)\bigr)^3\mathrm{BB}(t_0)}\right).
    """


@njit
def planck(nu_hz, t_k):
    """Return occupation number at frequency nu_hz and temperature t_k"""
    return 1 / (np.exp(c.H_OVER_K_B * nu_hz / t_k) - 1)


@njit
def compute_scalar_product(theta, phi, v):
    """Return the scalar (dot) product between a given direction and a velocity"""
    dx, dy, dz = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)

    return dx * v[0] + dy * v[1] + dz * v[2]


@njit
def calculate_beta(theta, phi, v_km_s):
    """Return a 2-tuple containing β·n and β"""
    beta_dot_n = compute_scalar_product(theta, phi, v_km_s) / c.C_LIGHT_KM_S
    beta = np.sqrt(v_km_s[0] ** 2 + v_km_s[1] ** 2 + v_km_s[2] ** 2) / c.C_LIGHT_KM_S

    return beta_dot_n, beta


@njit
def compute_dipole_for_one_sample_linear(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n = compute_scalar_product(theta, phi, v_km_s) / c.C_LIGHT_KM_S
    return t_cmb_k * beta_dot_n


@njit
def compute_dipole_for_one_sample_quadratic_exact(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)

    # Up to second order in beta, including second order in the expansion of
    # thermodynamic temperature. This is in true temperature, and
    # no boosting induced monopoles are added.
    return t_cmb_k * (beta_dot_n + beta_dot_n**2)


@njit
def compute_dipole_for_one_sample_total_exact(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)
    gamma = 1 / np.sqrt(1 - beta**2)

    return t_cmb_k / gamma / (1 - beta_dot_n) - t_cmb_k


@njit
def compute_dipole_for_one_sample_quadratic_from_lin_t(
    theta, phi, v_km_s, t_cmb_k, q_x
):
    # Up to second order in beta, including second order in the expansion of
    # thermodynamic temperature. This is in linearized thermodynamic temperature.
    # No boosting induced monopoles are added
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)
    return t_cmb_k * (beta_dot_n + q_x * beta_dot_n**2)


@njit
def compute_dipole_for_one_sample_total_from_lin_t(
    theta, phi, v_km_s, t_cmb_k, nu_hz, f_x, planck_t0
):
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)
    gamma = 1 / np.sqrt(1 - beta**2)

    planck_t = planck(nu_hz, t_cmb_k / gamma / (1 - beta_dot_n))

    return t_cmb_k / f_x * (planck_t / planck_t0 - 1)


@njit(parallel=True)
def add_dipole_for_one_detector(
    tod_det,
    theta_phi_det,
    velocity,
    t_cmb_k,
    nu_hz,
    f_x,
    q_x,
    dipole_type: DipoleType,
):
    if dipole_type == DipoleType.LINEAR:
        for i in prange(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_linear(
                theta=theta_phi_det[i, 0],
                phi=theta_phi_det[i, 1],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
            )
    elif dipole_type == DipoleType.QUADRATIC_EXACT:
        for i in prange(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_quadratic_exact(
                theta=theta_phi_det[i, 0],
                phi=theta_phi_det[i, 1],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
            )
    elif dipole_type == DipoleType.TOTAL_EXACT:
        for i in prange(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_total_exact(
                theta=theta_phi_det[i, 0],
                phi=theta_phi_det[i, 1],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
            )
    elif dipole_type == DipoleType.QUADRATIC_FROM_LIN_T:
        for i in prange(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_quadratic_from_lin_t(
                theta=theta_phi_det[i, 0],
                phi=theta_phi_det[i, 1],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
                q_x=q_x,
            )
    elif dipole_type == DipoleType.TOTAL_FROM_LIN_T:
        planck_t0 = planck(nu_hz, t_cmb_k)
        for i in prange(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_total_from_lin_t(
                theta=theta_phi_det[i, 0],
                phi=theta_phi_det[i, 1],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
                nu_hz=nu_hz,
                f_x=f_x,
                planck_t0=planck_t0,
            )
    else:
        print("Dipole Type not implemented!!!")


def add_dipole(
    tod,
    pointings,
    velocity,
    t_cmb_k: float,
    frequency_ghz: np.ndarray,  # e.g. central frequency of channel from
    # lbs.FreqChannelInfo.from_imo(url=…, imo=imo).bandcenter_ghz
    # using as url f"/releases/v1.0/satellite/{telescope}/{channel}/channel_info"
    dipole_type: DipoleType,
    pointings_dtype=np.float64,
):
    """Add the CMB dipole contribution to time-ordered data (TOD).

    Use `dipole_type` to specify which kind of approximation to use
    for the dipole component. The `pointings` argument must be a N×3 matrix containing
    the pointing information, where N is the size of the `tod` array. The `velocity`
    argument is usually computed through :func:`.spacecraft_pos_and_vel`.

    This function modifies the `tod` array by adding the contribution of the Cosmic
    Microwave Background (CMB) dipole, based on the provided pointing information,
    spacecraft velocity, and observation frequency.

    Parameters
    ----------
    tod : np.ndarray
        A 2D array representing the time-ordered data, with shape `(Ndetectors, Nsamples)`.

    pointings : np.ndarray or callable
        The pointing matrix, which can be:

        -   A numpy array of shape `(N_detectors, N_samples, 2)`, containing the pointing
            angles `(theta, phi)` for each detector and time sample.

        -   A callable function that returns the pointing information when called with
            `(detector_idx, pointings_dtype)`.

    velocity : np.ndarray
        A 2D array of shape `(N_samples, 3)`, representing the spacecraft velocity in
        three Cartesian coordinates over time.

    t_cmb_k : float
        The temperature of the CMB monopole in Kelvin.

    frequency_ghz : np.ndarray
        A 1D array of length `N_detectors`, containing the observation frequencies
        (band centers) in GHz for each detector.

    dipole_type : DipoleType
        Specifies the method used to compute the dipole contribution.

    pointings_dtype : dtype, optional
        Data type for pointings generated on the fly. If the pointing is passed or
        already precomputed this parameter is ineffective. Default is `np.float64`.

    Notes
    -----
    - The dipole contribution is computed individually for each detector using
      :func:`add_dipole_for_one_detector`.
    - If `pointings` is a numpy array, it must match the shape of `tod` along the
      first two dimensions.
    - The number of time samples in `tod` must match the length of `velocity`.

    Example
    -------

    Here is an example::

        add_dipole(
            tod=my_tod,
            pointings=my_pointings,
            velocity=my_velocity,
            t_cmb_k=2.725,
            frequency_ghz=np.array([30, 40, 70]),  # Example frequencies in GHz
            dipole_type=DipoleType.TOTAL_FROM_LIN_T,
        )

    """

    if type(pointings) is np.ndarray:
        assert tod.shape == pointings.shape[0:2]

    assert tod.shape[1] == velocity.shape[0]

    for detector_idx in range(tod.shape[0]):
        nu_hz = frequency_ghz[detector_idx] * 1e9  # freq in GHz
        # Note that x is a dimensionless parameter
        x = c.H_OVER_K_B * nu_hz / t_cmb_k

        f_x = x * np.exp(x) / (np.exp(x) - 1)

        q_x = 0.5 * x * (np.exp(x) + 1) / (np.exp(x) - 1)

        if type(pointings) is np.ndarray:
            theta_phi_det = pointings[detector_idx, :, :]
        else:
            theta_phi_det = pointings(detector_idx, pointings_dtype=pointings_dtype)[0][
                :, 0:2
            ]

        add_dipole_for_one_detector(
            tod_det=tod[detector_idx],
            theta_phi_det=theta_phi_det,
            velocity=velocity,
            t_cmb_k=t_cmb_k,
            nu_hz=nu_hz,
            f_x=f_x,
            q_x=q_x,
            dipole_type=dipole_type,
        )


def add_dipole_to_observations(
    observations: Union[Observation, List[Observation]],
    pos_and_vel: SpacecraftPositionAndVelocity,
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    t_cmb_k: float = c.T_CMB_K,
    dipole_type: DipoleType = DipoleType.TOTAL_FROM_LIN_T,
    frequency_ghz: Union[
        np.ndarray, None
    ] = None,  # e.g. central frequency of channel from
    component: str = "tod",
    pointings_dtype=np.float64,
):
    """Add the CMB dipole signal to the time-ordered data (TOD) stored
    in one or more `Observation` objects.

    This function acts as a wrapper around :func:`.add_dipole`,
    ensuring that the dipole signal is correctly computed and added to
    the TOD associated with `observations`.

    By default, the TOD is added to ``Observation.tod``. If you want
    to add it to some other field of the :class:`.Observation` class,
    use `component`.

    Args:
        observations (Union[Observation, List[Observation]]): A single
            :class:`Observation` instance or a list of `Observation` objects,
            whose TOD will be modified by adding the CMB dipole
            signal.

        pos_and_vel (:class:`.SpacecraftPositionAndVelocity`): An
            object providing spacecraft position and velocity
            information, which is necessary to compute the dipole
            contribution.

        pointings (Union[np.ndarray, List[np.ndarray], None]): If
            provided, this should be an array (or list of arrays)
            containing the pointing matrices for the observations. If
            `None`, the function will extract pointing matrices from
            the `Observation` objects.

        t_cmb_k (Optional[float]): The temperature of the cosmic
            microwave background (CMB) in Kelvin. Default is
            `c.T_CMB_K`.

        dipole_type (Optional[DipoleType]): The type of dipole to
            be added. Default is `DipoleType.TOTAL_FROM_LIN_T`.

        frequency_ghz (Optional[np.ndarray]): The observation
            frequency in GHz. If `None`, the function will use the
            `bandcenter_ghz` attribute from each `Observation`.

        component (Optional[str]): The name of the attribute in
            :class:`Observation` where the dipole signal should be
            stored. Default is ``"tod"``, but a different field (e.g.,
            ``"dipole_tod"``) can be specified.

        pointings_dtype (Optional[dtype]): Data type for pointings
            generated on the fly. If the pointing is passed or already
            precomputed this parameter is ineffective. Default is
            ``np.float64``.

    Note:
        If `pointings` is not provided, the function will extract
        pointing matrices from the `Observation` objects. If the
        pointing is generated on the fly, `pointings_dtype` specifies
        its type.

        The spacecraft velocity is interpolated over the timestamps of
        each observation.

    To add the dipole to the default TOD field::

        add_dipole_to_observations(sim.observations, pos_and_vel)


    To store the dipole in a separate component::

        for obs in sim.observations:
            obs.dipole_tod = np.zeros_like(obs.tod)

        add_dipole_to_observations(sim.observations, pos_and_vel, component="dipole_tod")

    """

    if pointings is None:
        if isinstance(observations, Observation):
            obs_list = [observations]
            if hasattr(observations, "pointing_matrix"):
                ptg_list = [observations.pointing_matrix[:, :, 0:2]]
            else:
                ptg_list = [observations.get_pointings]
        else:
            obs_list = observations
            ptg_list = []
            for ob in observations:
                if hasattr(ob, "pointing_matrix"):
                    ptg_list.append(ob.pointing_matrix[:, :, 0:2])
                else:
                    ptg_list.append(ob.get_pointings)
    else:
        if isinstance(observations, Observation):
            assert isinstance(pointings, np.ndarray), (
                "You must pass a list of observations *and* a list "
                + "of pointing matrices to add_dipole_to_observations"
            )
            obs_list = [observations]
            ptg_list = [pointings[:, :, 0:2]]
        else:
            assert isinstance(pointings, list), (
                "When you pass a list of observations to add_dipole_to_observations"
                + ", you must do the same for `pointings`"
            )
            assert len(observations) == len(pointings), (
                f"The list of observations has {len(observations)} elements, but "
                + f"the list of pointings has {len(pointings)} elements"
            )
            obs_list = observations
            ptg_list = [point[:, :, 0:2] for point in pointings]

    for cur_obs, cur_ptg in zip(obs_list, ptg_list):
        tod = getattr(cur_obs, component)

        # Alas, this allocates memory for the velocity vector! At the moment it is the
        # simplest implementation, but in the future we might want to inline the
        # interpolation code within "add_dipole" to save memory
        velocity = pos_and_vel.compute_velocities(
            time0=cur_obs.start_time,
            delta_time_s=cur_obs.get_delta_time().value,
            num_of_samples=tod.shape[1],
        )

        if frequency_ghz is None:
            frequency_ghz = cur_obs.bandcenter_ghz
        else:
            frequency_ghz = np.repeat(frequency_ghz, tod.shape[0])

        add_dipole(
            tod=tod,
            pointings=cur_ptg,
            velocity=velocity,
            t_cmb_k=t_cmb_k,
            frequency_ghz=frequency_ghz,
            dipole_type=dipole_type,
            pointings_dtype=pointings_dtype,
        )
