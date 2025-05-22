import litebird_sim as lbs
import numpy as np
from litebird_sim import mpi
import healpy as hp
from astropy.time import Time as astroTime
from litebird_sim.hwp_sys.hwp_sys import mueller_interpolation


def main(orbital_dipole, monopole, non_linearity, case, hwpss):
    start_time = astroTime("2026-01-01T00:00:00.000", format="isot")
    time_span_s = 365 * 24 * 3600
    nside = 128
    hwp_radpsec = lbs.IdealHWP(
        4.6 * 2 * np.pi / 60,
    ).ang_speed_radpsec

    lbs.PTEP_IMO_LOCATION = "schema.json"
    imo_version = "IMo_vReformationPlan_Option1M"
    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)
    channel = "MF1_140"

    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=0, imo=imo
    )

    comm = sim.mpi_comm

    channelinfo = lbs.FreqChannelInfo.from_imo(
        url=f"/releases/{imo_version}/LMHFT/{channel}/channel_info",
        imo=imo,
    )

    sim.set_scanning_strategy(
        imo_url=f"/releases/{imo_version}/Observation/Scanning_Strategy"
    )

    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            imo,
            f"/releases/{imo_version}/LMHFT/instrument_info",
        )
    )

    sim.set_hwp(lbs.IdealHWP(hwp_radpsec))

    dets = []
    for detidx in range(2):
        print(channelinfo.detector_names[detidx])
        det = lbs.DetectorInfo.from_imo(
            url=f"/releases/{imo_version}/LMHFT/{channel}/{channelinfo.detector_names[detidx]}/detector_info",
            imo=imo,
        )

        theta = det.pointing_theta_phi_psi_deg[0]

        det.mueller_hwp = {
            "0f": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
            "2f": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
            "4f": np.array(
                [
                    [0, 0, 0],
                    [mueller_interpolation(theta, "4f", 1, 0), 1, 1],
                    [mueller_interpolation(theta, "4f", 2, 0), 1, 1],
                ],
                dtype=np.float32,
            ),
        }

        det.g_one_over_k = -0.144
        det.amplitude_2f_k = 2.0
        det.optical_power_k = 1.5
        dets.append(det)

    # print(type(det.mueller_hwp))
    # print("sampling rate:", det.sampling_rate_hz)
    (obs,) = sim.create_observations(
        detectors=dets,
    )

    sim.prepare_pointings(append_to_report=False)

    if orbital_dipole:
        sim.compute_pos_and_vel()
        sim.add_dipole()

    if comm.rank == 0:
        Mbsparams = lbs.MbsParameters(
            make_cmb=True,
            seed_cmb=1234,
            make_noise=False,
            make_dipole=not (orbital_dipole),
            make_fg=True,
            fg_models=["pysm_synch_0", "pysm_dust_0", "pysm_freefree_1"],
            gaussian_smooth=True,
            bandpass_int=False,
            maps_in_ecliptic=True,
            nside=nside,
            units="K_CMB",
        )

        mbs = lbs.Mbs(simulation=sim, parameters=Mbsparams, channel_list=[channelinfo])

        input_maps = mbs.run_all()[0]["MF1_140"]

    else:
        input_maps = None

    if mpi.MPI_ENABLED:
        input_maps = comm.bcast(input_maps, root=0)

    if monopole:
        input_maps[0] += 2.7255

    hwp_sys = lbs.HwpSys(sim)

    hwp_sys.set_parameters(
        nside=nside,
        maps=input_maps,
        Channel=channelinfo,
        Mbsparams=Mbsparams,
        build_map_on_the_fly=True,
        apply_non_linearity=non_linearity,
        add_orbital_dipole=orbital_dipole,
        add_2f_hwpss=hwpss,
        comm=comm,
    )

    hwp_sys.fill_tod(
        observations=[obs],
        input_map_in_galactic=False,
    )

    stokes_parameters = ["I", "Q", "U"]
    for mp in range(3):
        hp.write_map(
            "input_maps" + case + stokes_parameters[mp] + ".fits",
            hwp_sys.maps[mp],
            overwrite=True,
        )

    output_maps = hwp_sys.make_map([obs])

    str_begn = "YENL" if non_linearity else "NONL"

    stokes_parameters = ["I", "Q", "U"]
    for mp in range(3):
        hp.write_map(
            str_begn + case + stokes_parameters[mp] + ".fits",
            output_maps[mp],
            overwrite=True,
        )


if __name__ == "__main__":
    main(
        orbital_dipole=True,
        monopole=False,
        non_linearity=True,
        case="_Case1_",
        hwpss=False,
    )
