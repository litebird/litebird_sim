# -*- encoding: utf-8 -*-
from collections import namedtuple
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
import numpy as np
from ducc0 import healpix
from astropy.time import Time

from typing import Union, List

import healpy  # We need healpy.read_map

import litebird_sim as lbs

from toast.todmap import OpMapMaker  # noqa: F401
from toast.tod.interval import Interval
import toast.mpi

from litebird_sim.coordinates import CoordinateSystem, rotate_coordinates_e2g

toast.mpi.use_mpi = lbs.MPI_ENABLED


@dataclass
class DestriperParameters:
    """Parameters used by the destriper to produce a map.

    The list of fields in this dataclass is the following:

    - ``nside``: the NSIDE parameter used to create the maps

    - ``coordinate_system``: an instance of the :class:`.CoordinateSystem` enum.
      It specifies if the map must be created in ecliptic (default) or
      galactic coordinates.

    - ``nnz``: number of components per pixel. The default is 3 (I/Q/U).

    - ``baseline_length_s``: length of the baseline for 1/f noise in seconds

    - ``iter_max``: maximum number of iterations

    - ``output_file_prefix``: prefix to be used for the filenames of the
      Healpix FITS maps saved in the output directory

    The following Boolean flags specify which maps should be returned
    by the function :func:`.destripe`:

    - ``return_hit_map``: return the hit map (number of hits per
      pixel)

    - ``return_binned_map``: return the binned map (i.e., the map with
      no baselines removed).

    - ``return_destriped_map``: return the destriped map. If pure
      white noise is present in the timelines, this should be the same
      as the binned map.

    - ``return_npp``: return the map of the white noise covariance per
      pixel. It contains the following fields: ``II``, ``IQ``, ``IU``,
      ``QQ``, ``QU``, and ``UU`` (in this order).

    - ``return_invnpp``: return the map of the inverse covariance per
      pixel. It contains the following fields: ``II``, ``IQ``, ``IU``,
      ``QQ``, ``QU``, and ``UU`` (in this order).

    - ``return_rcond``: return the map of condition numbers.

    The default is to only return the destriped map.

    """

    nside: int = 512
    coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic
    nnz: int = 3
    baseline_length_s: float = 60.0
    iter_max: int = 100
    output_file_prefix: str = "lbs_"
    return_hit_map: bool = False
    return_binned_map: bool = False
    return_destriped_map: bool = True
    return_npp: bool = False
    return_invnpp: bool = False
    return_rcond: bool = False


@dataclass
class DestriperResult:
    """Result of a call to :func:`.destripe`

    This dataclass has the following fields:

    - ``hit_map``: Healpix map containing the number of hit counts
      (integer values) per pixel

    - ``binned_map``: Healpix map containing the binned value for each pixel

    - ``destriped_map``: destriped Healpix mapmaker

    - ``npp``: covariance matrix elements for each pixel in the map

    - ``invnpp``: inverse of the covariance matrix element for each
      pixel in the map

    - ``rcond``: pixel condition number, stored as an Healpix map

    """

    hit_map: Any = None
    binned_map: Any = None
    destriped_map: Any = None
    npp: Any = None
    invnpp: Any = None
    rcond: Any = None
    coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic


class _Toast2FakeCache:
    "This class simulates a TOAST2 cache"

    def __init__(
        self,
        obs,
        pointings,
        nside,
        coordinates: CoordinateSystem,
    ):
        self.obs = obs

        self.keydict = {"timestamps": obs.get_times()}
        nsamples = len(self.keydict["timestamps"])

        self.keydict["flags"] = np.zeros(nsamples, dtype="uint8")

        if pointings is None:
            point = np.concatenate((obs.pointings, obs.psi[:, :, np.newaxis]), axis=2)
        else:
            point = pointings

        healpix_base = healpix.Healpix_Base(nside=nside, scheme="NEST")
        for (i, det) in enumerate(obs.name):
            if point[i].dtype == np.float64:
                curpnt = point[i]
            else:
                logging.warning(
                    "converting pointings for %s from %s to float64",
                    obs.name[i],
                    str(point[i].dtype),
                )
                curpnt = np.array(point[i], dtype=np.float64)

            if obs.tod[i].dtype == np.float64:
                self.keydict[f"signal_{det}"] = obs.tod[i]
            else:
                logging.warning(
                    "converting TODs for %s from %s to float64",
                    obs.name[i],
                    str(obs.tod[i].dtype),
                )
                self.keydict[f"signal_{det}"] = np.array(obs.tod[i], dtype=np.float64)

            theta_phi = curpnt[:, 0:2]
            polangle = curpnt[:, 2]

            if coordinates == CoordinateSystem.Galactic:
                theta_phi, polangle = rotate_coordinates_e2g(
                    pointings_ecl=theta_phi, pol_angle_ecl=polangle
                )
            elif coordinates == CoordinateSystem.Ecliptic:
                pass  # Do nothing, "theta_phi" and "polangle" are ok
            else:
                assert ValueError(
                    "unable to handle coordinate system {coordinates} in `destripe`"
                )

            self.keydict[f"pixels_{det}"] = healpix_base.ang2pix(theta_phi)
            self.keydict[f"weights_{det}"] = np.stack(
                (np.ones(nsamples), np.cos(2 * polangle), np.sin(2 * polangle))
            ).transpose()

    def keys(self):
        return self.keydict.keys()

    def reference(self, name):
        return self.keydict[name]

    def put(self, name, data, replace=False):
        if name is None:
            raise ValueError("Cache name cannot be None")

        if self.exists(name):
            if not replace:
                raise RuntimeError(f"Cache buffer {name} exists, but replace is False")

        self.keydict[name] = np.copy(data)

    def exists(self, name):
        return name in self.keydict

    def create(self, name, dtype, size):
        self.keydict[name] = np.empty(size, dtype=dtype)

    def destroy(self, name):
        del self.keydict[name]

    def __getitem__(self, item):
        return self.keydict[item]


class _Toast2FakeTod:
    "This class simulates a TOAST2 TOD"

    def __init__(
        self,
        obs,
        pointings,
        nside,
        coordinates: CoordinateSystem,
    ):
        self.obs = obs
        self.local_samples = (0, obs.tod[0].size)
        self.cache = _Toast2FakeCache(obs, pointings, nside, coordinates)

    def local_intervals(self, _):
        start_time = (
            self.obs.start_time.cxcsec
            if isinstance(self.obs.start_time, Time)
            else self.obs.start_time
        )
        return [
            Interval(
                start=start_time,
                stop=start_time + self.obs.sampling_rate_hz * self.obs.n_samples,
                first=0,
                last=self.obs.n_samples - 1,
            )
        ]

    def local_common_flags(self, _):
        return self.cache.reference("flags")

    @property
    def local_dets(self):
        return self.obs.name

    def local_flags(self, *args):
        return self.cache.reference("flags")

    def local_signal(self, det, name=None, **kwargs):
        if name is None:
            cachename = "{}_{}".format(self.SIGNAL_NAME, det)
            if not self.cache.exists(cachename):
                signal = self.read(detector=det, **kwargs)
                self.cache.put(cachename, signal)
        else:
            cachename = "{}_{}".format(name, det)
        return self.cache.reference(cachename)

    def local_times(self):
        return self.obs.get_times()


class _Toast2FakeData:
    "This class simulates a TOAST2 Data class"

    def __init__(
        self,
        obs,
        pointings,
        nside,
        coordinates: CoordinateSystem,
    ):
        if pointings is None:
            self.obs = [
                {"tod": _Toast2FakeTod(ob, None, nside, coordinates)} for ob in obs
            ]
        else:
            self.obs = [
                {"tod": _Toast2FakeTod(ob, po, nside, coordinates)}
                for ob, po in zip(obs, pointings)
            ]
        self.nside = nside
        if lbs.MPI_ENABLED:
            self.comm = toast.mpi.Comm(world=lbs.MPI_COMM_WORLD)
        else:
            CommWorld = namedtuple(
                "CommWorld", ["comm_world", "comm_group", "comm_rank", "comm_size"]
            )
            self.comm = CommWorld(
                comm_world=None, comm_group=None, comm_rank=0, comm_size=1
            )

        npix = 12 * (self.nside**2)
        self._metadata = {
            "pixels_npix": npix,
            "pixels_npix_submap": npix,
            "pixels_nsubmap": 1,
            "pixels_local_submaps": np.array([0], dtype=np.uint8),
        }

    def __getitem__(self, key):
        return self._metadata[key]

    def __setitem__(self, key, value):
        self._metadata[key] = value


def destripe_observations(
    observations,
    base_path: Path,
    params: DestriperParameters(),
    pointings: Union[List[np.ndarray], None] = None,
) -> DestriperResult:
    """Run the destriper on the observations in a TOD

    This function is a low-level wrapper around the TOAST destriper.
    For daily use, you should prefer the :func:`.destripe` function,
    which takes its parameters from :class:`.Simulation` object and
    is easier to call.

    This function runs the TOAST destriper on a set of `observations`
    (instances of the :class:`.Observation` class). The pointing
    information can be stored in the `observations` or passed through
    the variable `pointings`.

    The `params` parameter is an instance of the class
    :class:`.DestriperParameters`, and it specifies the way the
    destriper will be run and which kind of output is desired. The
    `base_path` parameter specifies where the Healpix FITS map will be
    saved. (TOAST's mapmaker cannot produce the maps in memory and
    must save them in FITS files.)

    """

    if pointings is not None:
        assert len(observations) == len(pointings), (
            f"The list of observations has {len(observations)}"
            + f" elements, but the list of pointings has {len(pointings)}"
        )

    data = _Toast2FakeData(
        obs=observations,
        pointings=pointings,
        nside=params.nside,
        coordinates=params.coordinate_system,
    )
    mapmaker = OpMapMaker(
        nside=params.nside,
        nnz=params.nnz,
        name="signal",
        outdir=base_path,
        outprefix=params.output_file_prefix,
        baseline_length=params.baseline_length_s,
        iter_max=params.iter_max,
        use_noise_prior=False,
    )

    mapmaker.exec(data)

    # Ensure that all the MPI processes are synchronized before
    # attempting to load the FITS files saved by OpMapMaker
    if lbs.MPI_ENABLED:
        lbs.MPI_COMM_WORLD.barrier()

    result = DestriperResult()

    result.coordinate_system = params.coordinate_system

    if params.return_hit_map:
        result.hit_map = healpy.read_map(
            base_path / (params.output_file_prefix + "hits.fits"),
            field=None,
            dtype=None,
        )

    if params.return_binned_map:
        result.binned_map = healpy.read_map(
            base_path / (params.output_file_prefix + "binned.fits"),
            field=None,
            dtype=None,
        )

    if params.return_destriped_map:
        result.destriped_map = healpy.read_map(
            base_path / (params.output_file_prefix + "destriped.fits"),
            field=None,
            dtype=None,
        )

    if params.return_npp:
        result.npp = healpy.read_map(
            base_path / (params.output_file_prefix + "npp.fits"), field=None, dtype=None
        )

    if params.return_invnpp:
        result.invnpp = healpy.read_map(
            base_path / (params.output_file_prefix + "invnpp.fits"),
            field=None,
            dtype=None,
        )

    if params.return_rcond:
        result.rcond = healpy.read_map(
            base_path / (params.output_file_prefix + "rcond.fits"),
            field=None,
            dtype=None,
        )

    return result


def destripe(
    sim,
    params=DestriperParameters(),
    pointings: Union[List[np.ndarray], None] = None,
) -> DestriperResult:
    """Run the destriper on a set of TODs.

    Run the TOAST destriper on time-ordered data, producing one or
    more Healpix maps. The `instrument` parameter must be an instance
    of the :class:`.Instrument` class and is used to convert pointings
    in Ecliptic coordinates, as it specifies the reference frame of
    the instrument hosting the detectors that produced the TODs. The
    `params` parameter is an instance of the
    :class:`.DestriperParameters` class, and it can be used to tune
    the way the TOAST destriper works.

    The function returns an instance of the class
    :class:`.DestriperResult`, which will contain only the maps
    specified by the ``make_*`` fields of the `params` parameter. (The
    default is only to return the destriped map in ``destriped_map``
    and set to ``None`` all the other fields.)

    The destriper will use *all* the observations in the `sim`
    parameter (an instance of the :class:`.Simulation` class); if you
    to run them only on a subset of observations (e.g., only the
    channels belonging to one frequency), you should use the function
    :func:`.destripe_observations` and pass it only the relevant
    :class:`.Observation` objects.

    """

    return destripe_observations(
        observations=sim.observations,
        base_path=sim.base_path,
        params=params,
        pointings=pointings,
    )
