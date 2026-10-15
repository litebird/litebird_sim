"""
GLS map-maker with correlated (1/f) noise using pysanepic

pysanepic does not depend on litebird_sim: this module converts the
observations into `pysanepic.DetectorData` (HEALPix pixels computed here,
orientation, HWP angle, TOD and the 1/f noise parameters used to simulate the
noise) and calls `pysanepic.make_maps`.
"""

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from ducc0.healpix import Healpix_Base

from litebird_sim.coordinates import CoordinateSystem
from litebird_sim.hwp import HWP
from litebird_sim.mpi import MPI_COMM_WORLD, MPI_ENABLED
from litebird_sim.noise import (
    _build_detector_groups,
    _normalize_rho,
    _validate_grouping,
)
from litebird_sim.observation_utilities import (
    _get_hwp_angle,
    _get_pointings_array,
    _normalize_observations_and_pointings,
)

from litebird_sim.utilities import resolve_nthreads

from .common import get_pol_efficiency

if TYPE_CHECKING:
    import pysanepic

_COORDINATES = {CoordinateSystem.Galactic: "G", CoordinateSystem.Ecliptic: "E"}


def _sum_components(obs, components, det_idx):
    """TOD of one detector summed over `components`, without a copy if there is one."""
    tod = getattr(obs, components[0])[det_idx]
    for c in components[1:]:
        tod = tod + getattr(obs, c)[det_idx]
    return tod


def _detector_flags(obs, names, det_idx):
    """Boolean flags of one detector, True where any of the `names` attributes
    (per detector or common to all) is nonzero; None without names."""
    if not names:
        return None
    out = np.zeros(obs.n_samples, dtype=bool)
    for name in names:
        f = np.asarray(getattr(obs, name))
        out |= (f[det_idx] if f.ndim == 2 else f) != 0
    return out


def _import_pysanepic(feature: str = "OneOverFNoise", version: str = "0.3.0"):
    try:
        import pysanepic
    except ImportError:
        raise ImportError(
            "Could not import `pysanepic`. Make sure that the package "
            "`pysanepic` is installed in the same environment as `litebird_sim`"
        )
    if not hasattr(pysanepic, feature):
        raise ImportError(
            f"pysanepic {pysanepic.__version__} is too old: version {version} or "
            "later is needed"
        )
    return pysanepic


def _detector_data(
    pysanepic,
    obs_list,
    ptg_list,
    hwp,
    nside,
    components,
    output_coordinate_system,
    pointings_dtype,
    nthreads,
    noise_model,
    flag_names,
    noise,
    correlation=None,
):
    """pysanepic.DetectorData for each detector of each observation, as a
    generator: pointings are computed for one detector at a time and turned
    into int32 pixels here, with the same pixelization as the other
    map-makers; pysanepic keeps only pixels and angles (no copy of the TODs).
    The noise is `noise[name]` if given, else the 1/f model of the detector.
    With `correlation` (the common-mode model of add_noise), the detectors of
    each group of an observation are yielded as one pysanepic.DetectorGroup."""
    coordinates = _COORDINATES[output_coordinate_system]
    hpx = Healpix_Base(nside, "RING")
    # DetectorData.name, used to average noise spectra, is in pysanepic >= 0.4.0
    named = "name" in {f.name for f in dataclasses.fields(pysanepic.DetectorData)}
    for obs, cur_ptg in zip(obs_list, ptg_list):
        hwp_angle = _get_hwp_angle(obs=obs, hwp=hwp, pointing_dtype=pointings_dtype)
        gamma = get_pol_efficiency(obs)
        net = np.broadcast_to(obs.net_ukrts, (obs.n_detectors,))
        if correlation is not None:
            yield from _detector_groups(
                pysanepic,
                obs,
                correlation,
                noise_model,
                net,
                _detector_data(
                    pysanepic,
                    [obs],
                    [cur_ptg],
                    hwp,
                    nside,
                    components,
                    output_coordinate_system,
                    pointings_dtype,
                    nthreads,
                    noise_model,
                    flag_names,
                    noise,
                ),
            )
            continue
        for det_idx in range(obs.n_detectors):
            name = str(obs.name[det_idx])
            ptg_det, hwp_angle = _get_pointings_array(
                detector_idx=det_idx,
                pointings=cur_ptg,
                hwp_angle=hwp_angle,
                output_coordinate_system=output_coordinate_system,
                pointings_dtype=pointings_dtype,
                nthreads=nthreads,
            )
            pointing = pysanepic.Pointing(
                pix=hpx.ang2pix(ptg_det[:, :2], nthreads=nthreads).astype(np.int32),
                nside=nside,
                psi=ptg_det[:, 2],
                coordinates=coordinates,
            )
            del ptg_det
            if noise is not None and name in noise:
                det_noise = noise[name]
            else:
                det_noise = pysanepic.OneOverFNoise(
                    net_ukrts=net[det_idx],
                    fknee_hz=obs.fknee_mhz[det_idx] / 1e3,
                    alpha=obs.alpha[det_idx],
                    fmin_hz=obs.fmin_hz[det_idx],
                    model=noise_model,
                )
            yield pysanepic.DetectorData(
                tod=_sum_components(obs, components, det_idx),
                pointing=pointing,
                hwp_angle=hwp_angle,
                sampling_rate_hz=obs.sampling_rate_hz,
                noise=det_noise,
                pol_angle_rad=obs.pol_angle_rad[det_idx],
                pol_efficiency=gamma[det_idx],
                flags=_detector_flags(obs, flag_names, det_idx),
                **({"name": name} if named else {}),
            )


def _detector_groups(pysanepic, obs, correlation, noise_model, net, data):
    """DetectorGroups of one observation for the common-mode model of
    add_noise: detector i of group g has n_i = sqrt(rho_i) c_g + sqrt(1 -
    rho_i) u_i, with u_i its own noise and c_g a stream with the noise
    parameters of the first detector of the group (1/f or white)."""
    if correlation.get("corr_matrix") is not None:
        raise ValueError(
            "make_sanepic_gls_map supports the common-mode correlation model, not corr_matrix"
        )
    if "groups" in correlation:
        groups = _validate_grouping(obs.n_detectors, correlation["groups"])
    else:
        groups = _build_detector_groups(obs, correlation.get("group_by"))
    rho = _normalize_rho(correlation.get("rho", 0.25), obs.n_detectors)
    if np.any(rho >= 1):
        raise ValueError("rho = 1 (no own noise) is not supported by pysanepic")
    common = correlation.get("common_mode_type", "one_over_f")
    data = list(data)  # this observation only
    for g in dict.fromkeys(groups):
        members = np.flatnonzero(groups == g)
        i0 = members[0]
        if common == "white":
            mode = pysanepic.OneOverFNoise(net_ukrts=net[i0])
        else:
            mode = pysanepic.OneOverFNoise(
                net_ukrts=net[i0],
                fknee_hz=obs.fknee_mhz[i0] / 1e3,
                alpha=obs.alpha[i0],
                fmin_hz=obs.fmin_hz[i0],
                model=noise_model,
            )
        dets = []
        for i in members:
            own = data[i].noise
            dets.append(
                dataclasses.replace(
                    data[i],
                    noise=dataclasses.replace(
                        own, net_ukrts=own.net_ukrts * np.sqrt(1 - rho[i])
                    ),
                )
            )
        yield pysanepic.DetectorGroup(
            detectors=dets,
            modes=[pysanepic.CommonMode(coupling=np.sqrt(rho[members]), noise=mode)],
        )


def make_sanepic_gls_map(
    nside: int,
    observations,
    pointings: npt.NDArray | list[npt.NDArray] | None = None,
    hwp: HWP | None = None,
    components: str | list[str] = "tod",
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    params: "pysanepic.GLSParameters | None" = None,
    pointings_dtype=np.float64,
    nthreads: int | None = None,
    noise_model: str = "toast",
    flags: str | list[str] | None = None,
    noise: dict | None = None,
    correlation: dict | None = None,
) -> "pysanepic.MapResult":
    """
    GLS map-maker with 1/f noise using pysanepic (SANEPIC algorithm).

    The noise covariance of each detector is modelled from the same
    parameters used by :func:`.add_noise_to_observations` (``net_ukrts``,
    ``fknee_mhz``, ``alpha``, ``fmin_hz``) and the same spectral shape
    (``noise_model``), and applied in Fourier space on chunks of
    ``params.chunk_s`` seconds.

    Parameters
    ----------
    nside : int
        Nside of the output map
    observations : Observation | list[Observation]
        Observations to use; with MPI, every process must call this function
    pointings : np.ndarray | list[np.ndarray], optional
        Pointings (θ, φ, ψ); by default they are taken from `observations`
    hwp : HWP, optional
        An instance of :class:`.HWP`; by default the one in the observations
    components : str | list[str], optional
        TOD components summed before map-making, by default "tod"
    output_coordinate_system : CoordinateSystem, optional
        Coordinate system of the output map, by default Galactic
    params : pysanepic.GLSParameters, optional
        Settings of the map-maker (chunk length, padding, preconditioner, PCG
        threshold, ...); by default ``pysanepic.GLSParameters()``. See the
        pysanepic documentation for the fields.
    pointings_dtype : dtype, optional
        dtype used to compute pointings on the fly, by default `np.float64`
    nthreads : int, optional
        Threads per process; by default resolved with :func:`.resolve_nthreads`
    noise_model : str, optional
        Shape of the 1/f spectrum, as the ``model`` argument of
        :func:`.add_noise_to_observations`: ``"toast"`` (default) or
        ``"keshner"`` (the one generated with ``engine="ducc"``). Use the
        model the noise was simulated with.
    flags : str | list[str], optional
        Names of the :class:`.Observation` attributes holding flags, either
        per detector ``(n_detectors, n_samples)`` (e.g. ``"local_flags"``) or
        common to all detectors ``(n_samples,)`` (e.g. ``"global_flags"``):
        samples with a nonzero flag in any of them are excluded from the maps
        (see ``flag_method`` and ``split_gap_s`` in ``pysanepic.GLSParameters``).
        By default no sample is excluded.
    noise : dict, optional
        Noise of the detectors, ``{detector name: pysanepic noise object}``,
        e.g. the spectra estimated by :func:`.estimate_sanepic_noise`; the
        detectors not in it use their 1/f parameters and ``noise_model``.
    correlation : dict, optional
        Noise correlated between detectors, with the common-mode model and
        the keys of :func:`.add_noise_to_observations` (``"group_by"`` or
        ``"groups"``, ``"rho"``, ``"common_mode_type"``): the detectors of
        each group of an observation are a ``pysanepic.DetectorGroup`` with
        one common mode. Correlations exist only between the detectors of
        the same observation: keep them together (e.g. ``n_blocks_det=1``
        and observations distributed over the processes). Needs pysanepic
        0.4.0 or later; not compatible with ``noise``.

    Returns
    -------
    pysanepic.MapResult
        Output maps (``maps``, ``hp.UNSEEN`` where not solved), ``hit_map``
        and the PCG convergence information
    """
    pysanepic = _import_pysanepic()
    if correlation is not None:
        pysanepic = _import_pysanepic("DetectorGroup", "0.4.0")
        if noise is not None:
            raise ValueError("correlation and noise cannot be used together")
    if noise_model not in ("toast", "keshner"):
        raise ValueError(
            f"Unknown noise model '{noise_model}': use 'toast' or 'keshner'"
        )
    flag_names = [flags] if isinstance(flags, str) else list(flags or [])
    if isinstance(components, str):
        components = [components]
    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )
    nthreads = resolve_nthreads(nthreads)
    data = _detector_data(
        pysanepic,
        obs_list,
        ptg_list,
        hwp,
        nside,
        components,
        output_coordinate_system,
        pointings_dtype,
        nthreads,
        noise_model,
        flag_names,
        noise,
        correlation,
    )

    comm = MPI_COMM_WORLD if MPI_ENABLED and MPI_COMM_WORLD.size > 1 else None
    return pysanepic.make_maps(
        data,
        nside,
        coordinates=_COORDINATES[output_coordinate_system],
        params=params,
        comm=comm,
        nthreads=nthreads,
    )


def estimate_sanepic_noise(
    result: "pysanepic.MapResult",
    observations,
    pointings: npt.NDArray | list[npt.NDArray] | None = None,
    hwp: HWP | None = None,
    components: str | list[str] = "tod",
    flags: str | list[str] | None = None,
    chunk_s: float = 86400.0,
    nbins: int = 60,
    fit: str | None = None,
    pointings_dtype=np.float64,
    nthreads: int | None = None,
) -> dict:
    """
    Noise spectrum of each detector from the residual of a pysanepic map.

    The residual between the TOD and the maps of `result` (e.g. a first call
    of :func:`.make_sanepic_gls_map`, with the same `components` and `flags`)
    is cut in chunks of `chunk_s` seconds and its periodogram is averaged
    over all the observations of each detector (also across MPI processes),
    in `nbins` logarithmic bins. See ``pysanepic.estimate_noise``.

    Returns ``{detector name: noise}``: a ``pysanepic.TabulatedNoise``, or
    with `fit` (``"toast"`` or ``"keshner"``) a fitted
    ``pysanepic.OneOverFNoise``; pass it to :func:`.make_sanepic_gls_map` as
    ``noise``. Needs pysanepic 0.4.0 or later.
    """
    pysanepic = _import_pysanepic("estimate_noise", "0.4.0")
    if isinstance(components, str):
        components = [components]
    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )
    nthreads = resolve_nthreads(nthreads)
    coordinate_system = {v: k for k, v in _COORDINATES.items()}[result.coordinates]
    data = _detector_data(
        pysanepic,
        obs_list,
        ptg_list,
        hwp,
        result.nside,
        components,
        coordinate_system,
        pointings_dtype,
        nthreads,
        "toast",  # not used: only the residual matters here
        [flags] if isinstance(flags, str) else list(flags or []),
        None,
    )
    comm = MPI_COMM_WORLD if MPI_ENABLED and MPI_COMM_WORLD.size > 1 else None
    return pysanepic.estimate_noise(
        data,
        result,
        chunk_s=chunk_s,
        nbins=nbins,
        fit=fit,
        comm=comm,
        nthreads=nthreads,
    )
