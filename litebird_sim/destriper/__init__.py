from dataclasses import dataclass
from typing import Any
import numpy as np
import healpy

import litebird_sim as lbs
from toast.todmap import OpMapMaker

if lbs.MPI_ENABLED:
    import toast.mpi


@dataclass
class DestriperParameters:
    """Parameters used by the destriper to produce a map.

    The list of fields in this dataclass is the following:

    - ``nnz``:

    - ``baseline_length``: number of consecutive samples in a 1/f noise
      baseline

    - ``iter_max``: maximum number of iterations

    - ``output_file_prefix``: prefix to be used for the filenames of the
      Healpix FITS maps saved in the output directory

    The following Boolean flags specify which maps should be returned
    by the function :func:`.destripe`:

    - ``return_hitmap``: return the hit map (number of hits per
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

    nnz: int = 3
    baseline_length: int = 100
    iter_max: int = 100
    output_file_prefix: str = "lbs_"
    return_hitmap: bool = False
    return_binned_map: bool = False
    return_destriped_map: bool = True
    return_npp: bool = False
    return_invnpp: bool = False
    return_rcond: bool = False


@dataclass
class DestriperResult:
    hitmap: Any = None
    binned_map: Any = None
    destriped_map: Any = None
    npp: Any = None
    invnpp: Any = None
    rcond: Any = None


class _Toast2FakeCache:
    def __init__(self, sim, obs, instr, nside):
        self.obs = obs

        self.keydict = {
            "timestamps": obs.get_times(),
        }
        nsamples = len(self.keydict["timestamps"])

        self.keydict["flags"] = np.zeros(nsamples, dtype="uint8")

        pointings = obs.get_pointings(
            sim.spin2ecliptic_quats,
            detector_quats=obs.quat,
            bore2spin_quat=instr.bore2spin_quat,
        )
        for (i, det) in enumerate(obs.name):
            curpnt = pointings[i]
            self.keydict[f"signal_{det}"] = obs.tod[i]
            self.keydict[f"pixels_{det}"] = healpy.ang2pix(
                nside,
                curpnt[:, 0],
                curpnt[:, 1],
                nest=True,
            )
            self.keydict[f"weights_{det}"] = np.stack(
                (
                    np.ones(nsamples),
                    np.cos(2 * curpnt[:, 2]),
                    np.sin(2 * curpnt[:, 2]),
                )
            ).transpose()

    def keys(self):
        return self.keydict.keys()

    def reference(self, name):
        return self.keydict[name]

    def put(self, name, data, **kwargs):
        self.keydict[name] = data

    def exists(self, name):
        return name in self.keydict

    def create(self, name, dtype, size):
        self.keydict[name] = np.empty(size, dtype=dtype)

    def destroy(self, name):
        del self.keydict[name]


class _Toast2FakeTod:
    def __init__(self, sim, obs, instr, nside):
        self.obs = obs
        self.local_samples = (0, obs.tod[0].size)
        self.cache = _Toast2FakeCache(sim, obs, instr, nside)

    def local_intervals(self, _):
        return [
            toast.tod.interval.Interval(
                start=self.obs.start_time,
                stop=self.obs.start_time
                + self.obs.sampling_rate_hz * self.obs.local_n_samples,
                first=0,
                last=self.obs.local_n_samples - 1,
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
    def __init__(self, sim, obs, instr, nside):
        self.sim = sim
        self.obs = [
            {
                "tod": _Toast2FakeTod(sim, x, instr, nside),
            }
            for x in obs
        ]
        self.instr = instr
        self.nside = nside
        if lbs.MPI_ENABLED:
            self.comm = toast.mpi.Comm(lbs.MPI_COMM_WORLD)
        else:
            self.comm = None

        self._metadata = {
            "pixels_npix": healpy.nside2npix(self.nside),
            "pixels_npix_submap": healpy.nside2npix(self.nside),
            "pixels_nsubmap": 1,
            "pixels_local_submaps": np.array([0], dtype=np.uint8),
        }

    def __getitem__(self, key):
        return self._metadata[key]

    def __setitem__(self, key, value):
        self._metadata[key] = value


def destripe(
    sim: lbs.Simulation,
    instrument,
    nside,
    params: DestriperParameters(),
):
    data = _Toast2FakeData(sim=sim, obs=sim.observations, instr=instrument, nside=nside)
    mapmaker = toast.todmap.OpMapMaker(
        nside=nside,
        nnz=params.nnz,
        name="signal",
        outdir=sim.base_path,
        outprefix=params.output_file_prefix,
        baseline_length=params.baseline_length,
        iter_max=params.iter_max,
        use_noise_prior=False,
    )

    mapmaker.exec(data)

    result = DestriperResult()

    if params.return_hitmap:
        result.hitmap = healpy.read_map(
            sim.base_path / (params.output_file_prefix + "_hits.fits"),
            field=None,
        )

    if params.return_binned_map:
        result.binned_map = healpy.read_map(
            sim.base_path / (params.output_file_prefix + "_binned.fits"),
            field=None,
        )

    if params.return_destriped_map:
        result.destriped_map = healpy.read_map(
            sim.base_path / (params.output_file_prefix + "_destriped.fits"),
            field=None,
        )

    if params.return_npp:
        result.npp = healpy.read_map(
            sim.base_path / (params.output_file_prefix + "_npp.fits"),
            field=None,
        )

    if params.return_invnpp:
        result.invnpp = healpy.read_map(
            sim.base_path / (params.output_file_prefix + "_invnpp.fits"),
            field=None,
        )

    if params.return_rcond:
        result.rcond = healpy.read_map(
            sim.base_path / (params.output_file_prefix + "_rcond.fits"),
            field=None,
        )

    return result
