"""
GLS map-maker with correlated (1/f) noise using pysanepic

pysanepic does not depend on litebird_sim: this module converts the
observations into `pysanepic.DetectorData` (pointings in the output
coordinates, as HEALPix pixels computed here or as angles, see
`pointing_form`; orientation, HWP angle, TOD and the 1/f noise parameters
used to simulate the noise) and calls `pysanepic.make_maps`.
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


def _sky_array(x):
    """A sky model or template for pysanepic (which does not depend on
    litebird_sim): an array of maps, or of alm with mmax = lmax, from an array
    or a SphericalHarmonics."""
    if hasattr(x, "lmax") and hasattr(x, "values"):
        if x.mmax != x.lmax:
            raise ValueError("pysanepic needs alm with mmax = lmax")
        if getattr(x, "nfreqs", None) not in (None, 1):
            raise ValueError("pysanepic needs alm at one frequency")
        return np.asarray(x.values)
    return np.asarray(x)


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
    dipole_component=None,
    angles=False,
    tod_templates=(),
):
    """pysanepic.DetectorData for each detector of each observation, as a
    generator: pointings are computed for one detector at a time and turned
    into int32 pixels here, with the same pixelization as the other
    map-makers (or, with `angles`, passed as angles in the output
    coordinates: pointing_form, needed by sky models given as alm); pysanepic keeps only pixels
    and angles (no copy of the TODs). With `dipole_component`, that TOD
    component is DetectorData.dipole (for calibration); the TOD components
    in `tod_templates` are DetectorData.templates.
    The noise is `noise[name]` if given, else the 1/f model of the detector.
    With `correlation` (the common-mode model of add_noise), the detectors of
    each group of an observation are yielded as one pysanepic.DetectorGroup,
    with the common modes `noise[names of the group]` if `noise` is given."""
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
                noise,
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
                    dipole_component=dipole_component,
                    angles=angles,
                    tod_templates=tod_templates,
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
            if angles:
                pointing = pysanepic.Pointing(
                    theta=ptg_det[:, 0].copy(),
                    phi=ptg_det[:, 1].copy(),
                    psi=ptg_det[:, 2].copy(),
                    coordinates=coordinates,
                )
            else:
                pointing = pysanepic.Pointing(
                    pix=hpx.ang2pix(ptg_det[:, :2], nthreads=nthreads).astype(np.int32),
                    nside=nside,
                    psi=ptg_det[:, 2],
                    coordinates=coordinates,
                )
            del ptg_det
            extra = {"name": name} if named else {}
            if dipole_component is not None:
                extra["dipole"] = getattr(obs, dipole_component)[det_idx]
            if tod_templates:
                extra["templates"] = {
                    k: getattr(obs, k)[det_idx] for k in tod_templates
                }
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
                **extra,
            )


def _detector_groups(pysanepic, obs, correlation, noise_model, net, noise, data):
    """DetectorGroups of one observation for the common-mode model of
    add_noise: detector i of group g has n_i = sqrt(rho_i) c_g + sqrt(1 -
    rho_i) u_i, with u_i its own noise and c_g a stream with the noise
    parameters of the first detector of the group (1/f or white). With
    `noise` (from estimate_sanepic_noise with `correlation`), the own noise
    of each detector and the modes of each group are taken from it."""
    if correlation.get("corr_matrix") is not None:
        raise ValueError(
            "make_sanepic_gls_map supports the common-mode correlation model, not corr_matrix"
        )
    if "groups" in correlation:
        groups = _validate_grouping(obs.n_detectors, correlation["groups"])
    else:
        groups = _build_detector_groups(obs, correlation.get("group_by"))
    rho = _normalize_rho(correlation.get("rho", 0.25), obs.n_detectors)
    if noise is None and np.any(rho >= 1):
        raise ValueError("rho = 1 (no own noise) is not supported by pysanepic")
    common = correlation.get("common_mode_type", "one_over_f")
    data = list(data)  # this observation only
    for g in dict.fromkeys(groups):
        members = np.flatnonzero(groups == g)
        if noise is not None:
            names = tuple(str(obs.name[i]) for i in members)
            if names not in noise:
                raise ValueError(
                    f"noise has no common modes for the group {names}: "
                    "estimate them with estimate_sanepic_noise(correlation=...)"
                )
            yield pysanepic.DetectorGroup(
                detectors=[data[i] for i in members], modes=noise[names]
            )
            continue
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
    calibration: "pysanepic.CalibrationParameters | None" = None,
    dipole_component: str | None = None,
    sky_model=None,
    templates: dict | None = None,
    template_covariance: bool = False,
    pointing_form: str = "auto",
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
        With ``correlation``, it must also have the common modes of each
        group, under the tuple of the names of its detectors, as returned
        by :func:`.estimate_sanepic_noise` with the same ``correlation``.
    correlation : dict, optional
        Noise correlated between detectors, with the common-mode model and
        the keys of :func:`.add_noise_to_observations` (``"group_by"`` or
        ``"groups"``, ``"rho"``, ``"common_mode_type"``): the detectors of
        each group of an observation are a ``pysanepic.DetectorGroup`` with
        one common mode. Correlations exist only between the detectors of
        the same observation: keep them together (e.g. ``n_blocks_det=1``
        and observations distributed over the processes). The modes are
        those of the model of :func:`.add_noise_to_observations`, or the
        estimated ones in ``noise``. Needs pysanepic 0.4.0 or later.
    calibration : pysanepic.CalibrationParameters, optional
        Fit one gain per detector and period (``gain_period_s``) together with
        the maps, using the dipole in `dipole_component` as calibrator (see the
        pysanepic documentation). The gains are in ``MapResult.gains``, one
        array per detector and observation of this process. Needs pysanepic
        0.5.0 or later.
    dipole_component : str, optional
        With `calibration`, the TOD component holding the dipole of each
        sample, e.g. filled by :meth:`.Simulation.add_dipole` with
        ``component="dipole"`` and the settings used for the data. It is
        subtracted from the calibrated TOD and must contain the orbital
        dipole, which sets the absolute calibration. To keep the solar dipole
        in the maps, put only the orbital dipole in it (an orbit with
        ``solar_velocity_km_s=0``) and the solar dipole in `sky_model`, as alm:
        its gradient within a pixel would otherwise leak through the 1/f
        weights. With a non-linear `dipole_type` the dipole of the total
        velocity is not the sum of the two: the difference (quadratic terms of
        the solar dipole and solar-orbital cross terms, a few uK) is not
        modelled.
    sky_model : np.ndarray or SphericalHarmonics, optional
        A model of the whole sky (I or I/Q/U in K, in the output coordinates,
        e.g. CMB + foregrounds at the frequency of the channel), as maps at
        `nside` or alm (mmax = lmax). Its TOD is subtracted and its binned map
        put back: given as alm it is interpolated at the pointing, so its
        structure below the pixel size does not leak through the 1/f weights.
        Needs pysanepic 0.5.0 or later.
    templates : dict, optional
        ``{name: pysanepic.Template}``, templates fitted with the maps (and
        the gains), with amplitudes per detector (also summing to zero, or
        common to all) or per period; in ``MapResult.templates``. A template
        with ``sky`` is a sky signal, maps or alm as `sky_model` (e.g.
        ``Template(sky=dfg_dnu, sum_zero=True)`` for the bandpass mismatch of
        the foregrounds, the derivative with respect to frequency at the
        frequency of the foregrounds in `sky_model`); one without ``sky`` is
        the TOD component of the observations with its name (e.g. a
        thermometer reading). Needs pysanepic 0.6.0 or later.
    template_covariance : bool, optional
        Also compute the covariance of the amplitudes per detector and common
        (``MapResult.template_covariance``): one more PCG solve per
        amplitude. False by default. Needs pysanepic 0.6.0 or later.
    pointing_form : str, optional
        How the pointings are passed to pysanepic, always in
        `output_coordinate_system`: ``"pixels"`` (HEALPix pixels at `nside`,
        computed here as in the other map-makers), ``"angles"`` (θ, φ, ψ,
        pixelized by pysanepic) or ``"auto"`` (default): angles if
        `sky_model` or some template is given as alm, which are
        interpolated at the pointing and need the angles, pixels otherwise.
        The maps are the same; pixels keep less memory per detector.

    Returns
    -------
    pysanepic.MapResult
        Output maps (``maps``, ``hp.UNSEEN`` where not solved), ``hit_map``,
        the PCG convergence information, and ``gains``, ``templates`` and
        ``template_covariance`` with calibration and templates
    """
    pysanepic = _import_pysanepic()
    if correlation is not None:
        pysanepic = _import_pysanepic("DetectorGroup", "0.4.0")
    extra = {}
    if calibration is not None:
        pysanepic = _import_pysanepic("CalibrationParameters", "0.5.0")
        if dipole_component is None:
            raise ValueError(
                "calibration needs dipole_component, the TOD component with the dipole"
            )
        extra["calibration"] = calibration
    if sky_model is not None:
        if not hasattr(pysanepic, "CalibrationParameters"):
            raise ImportError(
                f"pysanepic {pysanepic.__version__} is too old for sky_model: "
                "version 0.5.0 or later is needed"
            )
        extra["sky_model"] = _sky_array(sky_model)
    tod_templates = []
    if templates or template_covariance:
        pysanepic = _import_pysanepic("Template", "0.6.0")
        extra["templates"] = {
            k: t if t.sky is None else dataclasses.replace(t, sky=_sky_array(t.sky))
            for k, t in (templates or {}).items()
        }
        extra["template_covariance"] = template_covariance
        tod_templates = [k for k, t in extra["templates"].items() if t.sky is None]
    models = [
        extra.get("sky_model"),
        *(t.sky for t in extra.get("templates", {}).values()),
    ]
    alm = any(m is not None and np.iscomplexobj(m) for m in models)
    if pointing_form not in ("auto", "pixels", "angles"):
        raise ValueError(
            f"Unknown pointing_form '{pointing_form}': use 'auto', 'pixels' or 'angles'"
        )
    if pointing_form == "pixels" and alm:
        raise ValueError(
            "pointing_form='pixels' with sky_model or templates as alm: "
            "they are interpolated at the pointing and need the angles "
            "(use 'angles' or 'auto', or pass them as maps)"
        )
    angles = pointing_form == "angles" or (pointing_form == "auto" and alm)
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
        dipole_component=dipole_component if calibration is not None else None,
        angles=angles,
        tod_templates=tod_templates,
    )

    comm = MPI_COMM_WORLD if MPI_ENABLED and MPI_COMM_WORLD.size > 1 else None
    return pysanepic.make_maps(
        data,
        nside,
        coordinates=_COORDINATES[output_coordinate_system],
        params=params,
        comm=comm,
        nthreads=nthreads,
        **extra,
    )


def add_sanepic_beam_templates(
    observations,
    sky_model,
    pointings: npt.NDArray | list[npt.NDArray] | None = None,
    hwp: HWP | None = None,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    offsets: bool = True,
    ellipticity: bool = True,
    dtype=np.float64,
    pointings_dtype=np.float64,
    nthreads: int | None = None,
) -> dict:
    """
    Beam templates for :func:`.make_sanepic_gls_map`: pointing offsets and
    ellipticity of the beams, as TOD components of the observations.

    For each detector, the derivatives of `sky_model` along the axes of the
    detector at its pointing (``pysanepic.beam_tods``): the TODs
    ``offset_x``, ``offset_y`` (first derivatives) and ``ell_plus``,
    ``ell_cross`` (second derivatives) are stored in each observation, as
    arrays ``(n_detectors, n_samples)``, and fitted by
    :func:`.make_sanepic_gls_map` with the templates returned here. The
    amplitudes are the offsets of the beam centres [rad] and the changes of
    the second moments of the beams [rad²]; ``pysanepic.beam_parameters``
    turns them into beam axes. See the beam section of the pysanepic
    documentation for the frame of the detector and what is fitted. The FWHM
    is a sky template, ``pysanepic.fwhm_template``.

    Parameters
    ----------
    observations : Observation | list[Observation]
        Observations to use; with MPI, every process must call this function
    sky_model : np.ndarray or SphericalHarmonics
        The sky model through the mean beam of the detectors, alm (T or
        T/E/B, mmax = lmax) in `output_coordinate_system`: the `sky_model` of
        :func:`.make_sanepic_gls_map`
    pointings : np.ndarray | list[np.ndarray], optional
        Pointings (θ, φ, ψ); by default they are taken from `observations`
    hwp : HWP, optional
        An instance of :class:`.HWP`; by default the one in the observations
    output_coordinate_system : CoordinateSystem, optional
        Coordinate system of `sky_model` and of the maps, by default Galactic
    offsets, ellipticity : bool, optional
        Which templates to compute (by default both)
    dtype : dtype, optional
        dtype of the TOD components, by default `np.float64`
    pointings_dtype : dtype, optional
        dtype used to compute pointings on the fly, by default `np.float64`
    nthreads : int, optional
        Threads per process; by default resolved with :func:`.resolve_nthreads`

    Returns
    -------
    dict
        ``{name: pysanepic.Template}`` for the ``templates`` of
        :func:`.make_sanepic_gls_map`: one offset per detector, one
        ellipticity per detector, summing to zero without HWP (a common
        ellipticity is then degenerate with Q/U) and free with an HWP.
        Needs pysanepic 0.6.0 or later.
    """
    pysanepic = _import_pysanepic("beam_tods", "0.6.0")
    alm = _sky_array(sky_model)
    if not np.iscomplexobj(alm):
        raise ValueError("add_sanepic_beam_templates: sky_model must be alm")
    coordinates = _COORDINATES[output_coordinate_system]
    nthreads = resolve_nthreads(nthreads)
    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )
    noise = pysanepic.OneOverFNoise(net_ukrts=1.0)  # not used
    has_hwp = set()
    for obs, cur_ptg in zip(obs_list, ptg_list):
        hwp_angle = _get_hwp_angle(obs=obs, hwp=hwp, pointing_dtype=pointings_dtype)
        has_hwp.add(hwp_angle is not None)
        gamma = get_pol_efficiency(obs)
        out = {}
        for det_idx in range(obs.n_detectors):
            ptg_det, hwp_det = _get_pointings_array(
                detector_idx=det_idx,
                pointings=cur_ptg,
                hwp_angle=hwp_angle,
                output_coordinate_system=output_coordinate_system,
                pointings_dtype=pointings_dtype,
                nthreads=nthreads,
            )
            d = pysanepic.DetectorData(
                tod=np.broadcast_to(0.0, (obs.n_samples,)),
                pointing=pysanepic.Pointing(
                    theta=ptg_det[:, 0],
                    phi=ptg_det[:, 1],
                    psi=ptg_det[:, 2],
                    coordinates=coordinates,
                ),
                hwp_angle=hwp_det,
                sampling_rate_hz=obs.sampling_rate_hz,
                noise=noise,
                pol_angle_rad=obs.pol_angle_rad[det_idx],
                pol_efficiency=gamma[det_idx],
            )
            tods = pysanepic.beam_tods(
                d,
                alm,
                coordinates,
                offsets=offsets,
                ellipticity=ellipticity,
                nthreads=nthreads,
            )
            for k, v in tods.items():
                if k not in out:
                    out[k] = np.empty((obs.n_detectors, obs.n_samples), dtype=dtype)
                out[k][det_idx] = v
        for k, v in out.items():
            setattr(obs, k, v)
    if MPI_ENABLED and MPI_COMM_WORLD.size > 1:
        has_hwp = set().union(*MPI_COMM_WORLD.allgather(has_hwp))
    if len(has_hwp) > 1:
        raise ValueError(
            "add_sanepic_beam_templates: observations with and without HWP "
            "(the ellipticity constraints differ)"
        )
    has_hwp = True in has_hwp
    templates = {}
    if offsets:
        templates.update(offset_x=pysanepic.Template(), offset_y=pysanepic.Template())
    if ellipticity:
        templates.update(
            ell_plus=pysanepic.Template(sum_zero=not has_hwp),
            ell_cross=pysanepic.Template(sum_zero=not has_hwp),
        )
    return templates


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
    correlation: dict | None = None,
    n_modes: int = 1,
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

    With `correlation` (the groups of the common-mode model of
    :func:`.add_noise_to_observations`, see :func:`.make_sanepic_gls_map`),
    the cross-spectra of the detectors of each group are averaged over the
    observations and fitted with `n_modes` common modes plus the own noise
    of each detector: the result has the own noise under each name and the
    list of ``pysanepic.CommonMode`` under the tuple of the names of the
    group. Pass it to :func:`.make_sanepic_gls_map` with the same
    `correlation`. The parameters of the model (``rho``, the spectrum of
    the mode) are not used.
    """
    pysanepic = _import_pysanepic("estimate_noise", "0.4.0")
    extra = {}
    if correlation is not None:
        import inspect

        if "n_modes" not in inspect.signature(pysanepic.estimate_noise).parameters:
            raise ImportError(
                f"pysanepic {pysanepic.__version__} is too old to estimate common "
                "modes: version 0.4.0 or later is needed"
            )
        extra["n_modes"] = n_modes
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
        correlation,
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
        **extra,
    )
