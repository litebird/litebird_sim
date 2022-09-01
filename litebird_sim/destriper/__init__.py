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
from litebird_sim.mapping import DestriperParameters, DestriperResult

toast.mpi.use_mpi = lbs.MPI_ENABLED


class _Toast2FakeCache:
    "This class simulates a TOAST2 cache"

    def __init__(
        self,
        obs,
        pointings,
        nside,
        coordinates: CoordinateSystem,
        polarization: bool = True,
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

            if polarization:
                weights = np.stack(
                    (np.ones(nsamples), np.cos(2 * polangle), np.sin(2 * polangle))
                ).transpose()
            else:
                weights = np.ones(nsamples).reshape((-1, 1))

            self.keydict[f"weights_{det}"] = weights

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
        polarization: bool = True,
    ):
        self.obs = obs
        self.local_samples = (0, obs.tod[0].size)
        self.cache = _Toast2FakeCache(
            obs, pointings, nside, coordinates, polarization=polarization
        )

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
        polarization: bool = True,
    ):
        if pointings is None:
            self.obs = [
                {
                    "tod": _Toast2FakeTod(
                        ob, None, nside, coordinates, polarization=polarization
                    )
                }
                for ob in obs
            ]
        else:
            self.obs = [
                {
                    "tod": _Toast2FakeTod(
                        ob, po, nside, coordinates, polarization=polarization
                    )
                }
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

    polarization = params.nnz == 3

    data = _Toast2FakeData(
        obs=observations,
        pointings=pointings,
        nside=params.nside,
        coordinates=params.coordinate_system,
        polarization=polarization,
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
