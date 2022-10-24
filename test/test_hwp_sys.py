import litebird_sim as lbs
import numpy as np
import healpy as hp
from astropy.time import Time
import os


def test_hwp_sys():

    start_time = 0
    time_span_s = 10
    nside = 64
    sampling_hz = 1
    hwp_radpsec = 4.084_070_449_666_731

    sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s)

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
        start_time, time_span_s, delta_time_s=7200
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    detT = lbs.DetectorInfo(
        name="Boresight_detector_T",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    detB = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
    )

    (obs_h,) = sim.create_observations(detectors=[detT, detB])

    pointings = lbs.scanning.get_pointings(
        obs_h,
        spin2ecliptic_quats=spin2ecliptic_quats,
        detector_quats=[detT.quat, detB.quat],
        bore2spin_quat=instr.bore2spin_quat,
    )

    filepath = (
        os.path.dirname(__file__).strip("test")
        + "litebird_sim/out_of_band_analysis/examples/MFT_100_h_beta_z.txt"
    )
    mft = np.loadtxt(filepath)

    h1 = mft[:, 1]
    h2 = mft[:, 2]
    beta = mft[:, 3]
    z1 = mft[:, 4]
    z2 = mft[:, 5]

    par = {
        "hwp_sys": {
            "band_filename": filepath,
            "band_filename_solver": filepath,  # same as tod parameters
        }
    }

    sim.parameter_file = par  # setting the parameter file
    hwp_sys = lbs.HwpSys(sim)

    Mbsparams = lbs.MbsParameters(
        make_cmb=True,
        make_fg=True,
        fg_models=["pysm_synch_0", "pysm_freefree_1", "pysm_dust_0"],
        bandpass_int=True,
        maps_in_ecliptic=True,
        seed_cmb=1234,
        nside=nside,
    )

    hwp_sys = lbs.HwpSys(sim)

    hwp_sys.set_parameters(
        integrate_in_band=True,
        integrate_in_band_solver=True,
        correct_in_solver=True,
        built_map_on_the_fly=False,
        nside=nside,
        Mbsparams=Mbsparams,
    )

    np.testing.assert_equal(hwp_sys.h1, h1)
    np.testing.assert_equal(hwp_sys.h1s, h1)
    np.testing.assert_equal(hwp_sys.h2, h2)
    np.testing.assert_equal(hwp_sys.h2s, h2)
    np.testing.assert_equal(np.cos(hwp_sys.beta), np.cos(np.deg2rad(beta)))
    np.testing.assert_equal(np.cos(hwp_sys.betas), np.cos(np.deg2rad(beta)))
    np.testing.assert_equal(hwp_sys.z1, z1)
    np.testing.assert_equal(hwp_sys.z1s, z1)
    np.testing.assert_equal(hwp_sys.z2, z2)
    np.testing.assert_equal(hwp_sys.z2s, z2)

    hwp_sys.fill_tod(obs_h, pointings, hwp_radpsec)

    reference = np.array(
        [
            [
                3.8648595e-05,
                -3.9369585e-05,
                -3.7868820e-05,
                -3.8603906e-05,
                -3.8943563e-05,
                -3.8293791e-05,
                -3.8582919e-05,
                -3.8396141e-05,
                -3.9056486e-05,
                -3.8146281e-05,
            ],
            [
                3.9318311e-05,
                -3.8004528e-05,
                -3.8656683e-05,
                -3.8627029e-05,
                -3.8694950e-05,
                -3.8384474e-05,
                -3.8288104e-05,
                -3.9314618e-05,
                -3.7936523e-05,
                -3.8455721e-05,
            ],
        ],
        dtype=np.float32,
    )

    np.testing.assert_equal(obs_h.tod, reference)
