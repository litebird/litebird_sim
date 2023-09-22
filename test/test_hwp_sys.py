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

    sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s, random_seed=0)

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

    detBT = lbs.DetectorInfo(
        name="Boresight_detector_T",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    detBB = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
    )

    det165 = lbs.DetectorInfo(
        name="not_boresight_detector_165",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[-0.07962602, 0.07427495, 0.98554952, 0.12975006],
    )

    det105 = lbs.DetectorInfo(
        name="not_boresight_detector_105",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.00924192, -0.10162824, -0.78921165, 0.6055834],
    )

    (obs_boresight,) = sim.create_observations(detectors=[detBT, detBB])

    (pointings_b,) = lbs.get_pointings_for_observations(
        sim.observations,
        spin2ecliptic_quats=spin2ecliptic_quats,
        bore2spin_quat=instr.bore2spin_quat,
        hwp=None,
        store_pointings_in_obs=True,
    )

    (obs_no_boresight,) = sim.create_observations(detectors=[det165, det105])

    (pointings_nob,) = lbs.get_pointings_for_observations(
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
    hwp_sys = lbs.HwpSys(sim)

    Mbsparams = lbs.MbsParameters(
        make_cmb=True,
        make_fg=True,
        fg_models=["pysm_synch_1", "pysm_freefree_1", "pysm_dust_1", "pysm_ame_1"],
        bandpass_int=True,
        maps_in_ecliptic=False,
        seed_cmb=1234,
        nside=nside,
    )

    hwp_sys = lbs.HwpSys(sim)

    hwp_sys.set_parameters(
        mueller_or_jones="jones",
        integrate_in_band=True,
        integrate_in_band_solver=True,
        correct_in_solver=True,
        built_map_on_the_fly=False,
        nside=nside,
        Mbsparams=Mbsparams,
    )

    np.testing.assert_equal(hwp_sys.bandpass_profile, hwp_sys.bandpass_profile_solver)
    np.testing.assert_equal(hwp_sys.freqs, hwp_sys.freqs_solver)

    # testing if code works also with list of obs of the same channel
    hwp_sys.fill_tod(
        obs=[obs_boresight, obs_no_boresight], hwp_radpsec=hwp_radpsec
    )  # pointings = pointings,

    reference_b = np.array(
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

    reference_nob = np.array(
        [
            [
                1.6051326e-05,
                1.6835435e-05,
                1.5598331e-05,
                1.6891758e-05,
                1.6264255e-05,
                1.5769723e-05,
                1.7102797e-05,
                1.5715126e-05,
                1.6552618e-05,
                1.6356096e-05,
            ],
            [
                -4.7307349e-05,
                -4.8946167e-05,
                -4.7057129e-05,
                -4.8257643e-05,
                -4.7521422e-05,
                -4.8152131e-05,
                -4.8154547e-05,
                -4.6702909e-05,
                -4.9267896e-05,
                -4.7250538e-05,
            ],
        ],
        dtype=np.float32,
    )

    np.testing.assert_equal(obs_boresight.tod, reference_b)
    np.testing.assert_equal(obs_no_boresight.tod, reference_nob)
