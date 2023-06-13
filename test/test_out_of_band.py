import litebird_sim as lbs
import numpy as np
import healpy as hp
from astropy.time import Time
import os


def test_out_of_band():

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

    (obs_o,) = sim.create_observations(detectors=[detT, detB])

    (pointings,) = lbs.get_pointings_for_observations(
        sim.observations,
        spin2ecliptic_quats=spin2ecliptic_quats,
        bore2spin_quat=instr.bore2spin_quat,
        hwp=None,
        store_pointings_in_obs=True,
    )

    filepath = (
        os.path.dirname(__file__).strip("test")
        + "litebird_sim/out_of_band_analysis/examples/MFT_100_h_beta_z.txt"
    )
    mft = np.loadtxt(filepath)

    nu = mft[:, 0]
    h1 = mft[:, 1]
    h2 = mft[:, 2]
    beta = mft[:, 3]
    z1 = mft[:, 4]
    z2 = mft[:, 5]

    par = {
        "hwp_sys": {
            "band_filename": filepath,
            "band_filename_solver": filepath,  # same as tod parameters
            "bandpass": {
                "band_type": "top-hat",
                "band_low_edge": nu[0],
                "band_high_edge": nu[-1],
                "bandcenter_ghz": 100,
            },
            "bandpass_solver": {
                "band_type": "top-hat",
                "band_low_edge": nu[0],
                "band_high_edge": nu[-1],
                "bandcenter_ghz": 100,
            },
            "include_beam_throughput": False,
        }
    }

    sim.parameter_file = par  # setting the parameter file
    hwp_sys_band = lbs.HwpSysAndBandpass(sim)

    Mbsparams = lbs.MbsParameters(
        make_cmb=True,
        make_fg=True,
        fg_models=["pysm_synch_1", "pysm_freefree_1", "pysm_dust_1", "pysm_ame_1"],
        bandpass_int=True,
        maps_in_ecliptic=False,
        seed_cmb=1234,
        nside=nside,
    )

    hwp_sys_band.set_parameters(
        integrate_in_band=True,
        integrate_in_band_solver=True,
        correct_in_solver=True,
        built_map_on_the_fly=False,
        nside=nside,
        Mbsparams=Mbsparams,
    )

    hwp_sys_band.fill_tod(obs=obs_o, hwp_radpsec=hwp_radpsec)  # pointings = pointings,

    np.testing.assert_equal(hwp_sys_band.h1, h1)
    np.testing.assert_equal(hwp_sys_band.h1s, h1)
    np.testing.assert_equal(hwp_sys_band.h2, h2)
    np.testing.assert_equal(hwp_sys_band.h2s, h2)
    np.testing.assert_equal(hwp_sys_band.cbeta, np.cos(np.deg2rad(beta)))
    np.testing.assert_equal(hwp_sys_band.cbetas, np.cos(np.deg2rad(beta)))
    np.testing.assert_equal(hwp_sys_band.z1, z1)
    np.testing.assert_equal(hwp_sys_band.z1s, z1)
    np.testing.assert_equal(hwp_sys_band.z2, z2)
    np.testing.assert_equal(hwp_sys_band.z2s, z2)

    reference = np.array(
        [
            [
                3.0560230e-05,
                2.9122459e-05,
                2.9265628e-05,
                3.0336547e-05,
                -1.9575957e-05,
                -1.8873492e-05,
                -1.9916168e-05,
                -1.9199055e-05,
                -1.9401938e-05,
                -1.9470222e-05,
            ],
            [
                2.8618011e-05,
                3.0030897e-05,
                2.9890767e-05,
                2.8842858e-05,
                -1.9244833e-05,
                -1.9962308e-05,
                -1.8914921e-05,
                -1.9619934e-05,
                -1.9429232e-05,
                -1.9365529e-05,
            ],
        ],
        dtype=np.float32,
    )

    np.testing.assert_equal(obs_o.tod, reference)
