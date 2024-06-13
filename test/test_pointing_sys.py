# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs
from uuid import UUID

start_time = 0
time_span_s = 10
sampling_hz = 1


imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)

def test_get_detector_orientation():
    uuids = [
        UUID('f77095ee-1444-4be9-926f-be1faa3c40e4'),
        UUID('054f432b-bec5-47f9-97a9-c35ce4b18d56'),
        UUID('8c64faa0-eae6-4123-847f-6f3b15211dde'),
        UUID('4d3ef213-b72f-4b5c-8cd3-282985ec268d'),
        ]
    reference_orient = [0.0, 90.0, 45.0, 135.0]
    for i, uuid in enumerate(uuids):
        det = lbs.DetectorInfo.from_imo(imo, uuid)
        orient = lbs.get_detector_orientation(det)
        assert np.allclose(orient, np.deg2rad(reference_orient[i]))

def gen_simulation_and_dets():
    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=None
    )

    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)
    imo_version = "vPTEP"
    telescope = "LFT"
    channel = "L4-140"
    detlist = ["000_001_017_QB_140_T", "000_001_017_QB_140_B"]
    dets = []  # type: List[lbs.DetectorInfo]
    for n_det in detlist:
        det = lbs.DetectorInfo.from_imo(
            url=f"/releases/{imo_version}/satellite/{telescope}/{channel}/{n_det}/detector_info",
            imo=imo,
        )
        det.sampling_rate_hz = sampling_hz
        dets.append(det)

    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            imo,
            f"/releases/{imo_version}/satellite/{telescope}/instrument_info",
        )
    )

    sim.set_scanning_strategy(
            imo_url=f"/releases/{imo_version}/satellite/scanning_parameters/",
            delta_time_s=1.0/sampling_hz,
        )

    sim.set_hwp(
        lbs.IdealHWP(sim.instrument.hwp_rpm * 2 * np.pi / 60))

    return sim, dets

def test_PointingSys_add_single_offset_to_FP():
    sim, dets = gen_simulation_and_dets()

    pointing_sys = lbs.PointingSys(sim.instrument, dets)
    single_offset = np.deg2rad(1.0)
    axis = 'x'
    pointing_sys.focalplane.add_offset(single_offset, axis)

    sim.create_observations(detectors=dets)


    lbs.prepare_pointings(
        sim.observations,
        sim.instrument,
        sim.spin2ecliptic_quats,
        hwp=sim.hwp
        )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(det_idx, pointings_dtype=np.float32)
            pointings_list.append(pointings)

    pointing_reference = [
        np.array([
            [ 1.67551601e+00, -3.48855772e-18,  1.62564003e+00],
            [ 1.67550623e+00,  4.14894568e-03,  1.62991035e+00],
            [ 1.67547679e+00,  8.29779729e-03,  1.63418043e+00],
            [ 1.67542768e+00,  1.24464575e-02,  1.63845026e+00],
            [ 1.67535901e+00,  1.65948346e-02,  1.64271998e+00],
            [ 1.67527068e+00,  2.07428336e-02,  1.64698911e+00],
            [ 1.67516267e+00,  2.48903558e-02,  1.65125787e+00],
            [ 1.67503512e+00,  2.90373117e-02,  1.65552604e+00],
            [ 1.67488790e+00,  3.31836045e-02,  1.65979338e+00],
            [ 1.67472112e+00,  3.73291373e-02,  1.66406012e+00]
            ]),
        np.array([
            [1.67551601e+00, 0.00000000e+00, 1.76343659e-04],
            [1.67550623e+00, 4.14894568e-03, 4.44655120e-03],
            [1.67547679e+00, 8.29779729e-03, 8.71666055e-03],
            [1.67542768e+00, 1.24464575e-02, 1.29865687e-02],
            [1.67535901e+00, 1.65948346e-02, 1.72561817e-02],
            [1.67527068e+00, 2.07428336e-02, 2.15253998e-02],
            [1.67516267e+00, 2.48903558e-02, 2.57941186e-02],
            [1.67503512e+00, 2.90373117e-02, 3.00622415e-02],
            [1.67488790e+00, 3.31836045e-02, 3.43296826e-02],
            [1.67472112e+00, 3.73291373e-02, 3.85963246e-02]
            ])
        ]
    np.testing.assert_allclose(pointings_list, pointing_reference)

def test_PointingSys_add_multiple_offsets_to_FP():
    sim, dets = gen_simulation_and_dets()

    pointing_sys = lbs.PointingSys(sim.instrument, dets)
    multiple_offsets = [np.deg2rad(1.0), np.deg2rad(0.5)]
    axis = 'x'
    pointing_sys.focalplane.add_offset(multiple_offsets, axis)

    sim.create_observations(detectors=dets)


    lbs.prepare_pointings(
        sim.observations,
        sim.instrument,
        sim.spin2ecliptic_quats,
        hwp=sim.hwp
        )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(det_idx, pointings_dtype=np.float32)
            pointings_list.append(pointings)

    pointing_reference = [
        np.array([
            [ 1.67551601e+00, -3.48855772e-18,  1.62564003e+00],
            [ 1.67550623e+00,  4.14894568e-03,  1.62991035e+00],
            [ 1.67547679e+00,  8.29779729e-03,  1.63418043e+00],
            [ 1.67542768e+00,  1.24464575e-02,  1.63845026e+00],
            [ 1.67535901e+00,  1.65948346e-02,  1.64271998e+00],
            [ 1.67527068e+00,  2.07428336e-02,  1.64698911e+00],
            [ 1.67516267e+00,  2.48903558e-02,  1.65125787e+00],
            [ 1.67503512e+00,  2.90373117e-02,  1.65552604e+00],
            [ 1.67488790e+00,  3.31836045e-02,  1.65979338e+00],
            [ 1.67472112e+00,  3.73291373e-02,  1.66406012e+00]
        ]),
        np.array([
            [1.6667894e+00, 0.0000000e+00, 1.7634366e-04],
            [1.6667796e+00, 4.1115093e-03, 4.4428008e-03],
            [1.6667504e+00, 8.2229255e-03, 8.7091606e-03],
            [1.6667018e+00, 1.2334155e-02, 1.2975328e-02],
            [1.6666336e+00, 1.6445108e-02, 1.7241204e-02],
            [1.6665460e+00, 2.0555684e-02, 2.1506689e-02],
            [1.6664389e+00, 2.4665799e-02, 2.5771700e-02],
            [1.6663123e+00, 2.8775357e-02, 3.0036123e-02],
            [1.6661663e+00, 3.2884259e-02, 3.4299873e-02],
            [1.6660008e+00, 3.6992423e-02, 3.8562849e-02]
            ])
        ]
    np.testing.assert_allclose(pointings_list, pointing_reference)

def test_PointingSys_add_multiple_offsets_to_FP():
    sim, dets = gen_simulation_and_dets()

    pointing_sys = lbs.PointingSys(sim.instrument, dets)
    multiple_offsets = [np.deg2rad(1.0), np.deg2rad(0.5)]
    axis = 'x'
    pointing_sys.focalplane.add_offset(multiple_offsets, axis)

    sim.create_observations(detectors=dets)


    lbs.prepare_pointings(
        sim.observations,
        sim.instrument,
        sim.spin2ecliptic_quats,
        hwp=sim.hwp
        )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(det_idx, pointings_dtype=np.float32)
            pointings_list.append(pointings)

    pointing_reference = [
        np.array([
            [ 1.67551601e+00, -3.48855772e-18,  1.62564003e+00],
            [ 1.67550623e+00,  4.14894568e-03,  1.62991035e+00],
            [ 1.67547679e+00,  8.29779729e-03,  1.63418043e+00],
            [ 1.67542768e+00,  1.24464575e-02,  1.63845026e+00],
            [ 1.67535901e+00,  1.65948346e-02,  1.64271998e+00],
            [ 1.67527068e+00,  2.07428336e-02,  1.64698911e+00],
            [ 1.67516267e+00,  2.48903558e-02,  1.65125787e+00],
            [ 1.67503512e+00,  2.90373117e-02,  1.65552604e+00],
            [ 1.67488790e+00,  3.31836045e-02,  1.65979338e+00],
            [ 1.67472112e+00,  3.73291373e-02,  1.66406012e+00]
        ]),
        np.array([
            [1.6667894e+00, 0.0000000e+00, 1.7634366e-04],
            [1.6667796e+00, 4.1115093e-03, 4.4428008e-03],
            [1.6667504e+00, 8.2229255e-03, 8.7091606e-03],
            [1.6667018e+00, 1.2334155e-02, 1.2975328e-02],
            [1.6666336e+00, 1.6445108e-02, 1.7241204e-02],
            [1.6665460e+00, 2.0555684e-02, 2.1506689e-02],
            [1.6664389e+00, 2.4665799e-02, 2.5771700e-02],
            [1.6663123e+00, 2.8775357e-02, 3.0036123e-02],
            [1.6661663e+00, 3.2884259e-02, 3.4299873e-02],
            [1.6660008e+00, 3.6992423e-02, 3.8562849e-02]
            ])
        ]
    np.testing.assert_allclose(pointings_list, pointing_reference)

def test_PointingSys_add_uncommon_disturb_to_FP():
    sim, dets = gen_simulation_and_dets()

    nquats = sim.spin2ecliptic_quats.quats.shape[0]
    noise_rad_matrix = np.zeros([len(dets), nquats])
    sigmas = [1.0, 0.5]
    sim.init_random(random_seed=12_345)
    for i in range(len(dets)):
        lbs.add_white_noise(noise_rad_matrix[i,:], sigma=np.deg2rad(sigmas[i]), random=sim.random)

    pointing_sys = lbs.PointingSys(sim.instrument, dets)
    axis = 'x'
    pointing_sys.focalplane.add_disturb(start_time, sampling_hz, noise_rad_matrix, axis)

    sim.create_observations(detectors=dets)


    lbs.prepare_pointings(
        sim.observations,
        sim.instrument,
        sim.spin2ecliptic_quats,
        hwp=sim.hwp
        )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(det_idx, pointings_dtype=np.float32)
            pointings_list.append(pointings)

    pointing_reference = [
        np.array([
            [1.6892608 , 0.        , 1.62564   ],
            [1.6495248 , 0.00403767, 1.6299001 ],
            [1.6769145 , 0.00831014, 1.6341817 ],
            [1.6501241 , 0.01212131, 1.6384205 ],
            [1.6484656 , 0.01613412, 1.642678  ],
            [1.662634  , 0.02047188, 1.6469626 ],
            [1.6509066 , 0.02426689, 1.6512004 ],
            [1.6537316 , 0.0283983 , 1.6554663 ],
            [1.6610764 , 0.0327098 , 1.6597475 ],
            [1.687686  , 0.03783091, 1.6641153 ]
            ], dtype=np.float32),
        np.array([
            [1.64833677e+00, 0.00000000e+00, 1.76343645e-04],
            [1.66518915e+00, 4.10469249e-03, 4.44215210e-03],
            [1.66194797e+00, 8.18177592e-03, 8.70531611e-03],
            [1.66171539e+00, 1.22700669e-02, 1.29693495e-02],
            [1.66601717e+00, 1.64345391e-02, 1.72401983e-02],
            [1.65120137e+00, 2.02272832e-02, 2.14778036e-02],
            [1.66061127e+00, 2.45160032e-02, 2.57578306e-02],
            [1.66204917e+00, 2.86474898e-02, 3.00242025e-02],
            [1.65736437e+00, 3.25826705e-02, 3.42724770e-02],
            [1.65249014e+00, 3.64718214e-02, 3.85168679e-02]],
                 dtype=np.float32)
        ]
    np.testing.assert_allclose(pointings_list, pointing_reference, atol=1e-8)

def test_PointingSys_add_common_disturb_to_FP():
    sim, dets = gen_simulation_and_dets()

    nquats = sim.spin2ecliptic_quats.quats.shape[0]
    noise_rad_1d_array = np.zeros(nquats)

    sim.init_random(random_seed=12_345)
    lbs.add_white_noise(noise_rad_1d_array, sigma=np.deg2rad(1), random=sim.random)

    pointing_sys = lbs.PointingSys(sim.instrument, dets)
    axis = 'x'
    pointing_sys.focalplane.add_disturb(start_time, sampling_hz, noise_rad_1d_array, axis)

    sim.create_observations(detectors=dets)


    lbs.prepare_pointings(
        sim.observations,
        sim.instrument,
        sim.spin2ecliptic_quats,
        hwp=sim.hwp
        )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(det_idx, pointings_dtype=np.float32)
            pointings_list.append(pointings)

    pointing_reference = [
        np.array([
            [1.6892608 , 0.        , 1.62564   ],
            [1.6495248 , 0.00403767, 1.6299001 ],
            [1.6769145 , 0.00831014, 1.6341817 ],
            [1.6501241 , 0.01212131, 1.6384205 ],
            [1.6484656 , 0.01613412, 1.642678  ],
            [1.662634  , 0.02047188, 1.6469626 ],
            [1.6509066 , 0.02426689, 1.6512004 ],
            [1.6537316 , 0.0283983 , 1.6554663 ],
            [1.6610764 , 0.0327098 , 1.6597475 ],
            [1.687686  , 0.03783091, 1.6641153 ]], dtype=np.float32),
        np.array([
            [1.6892608e+00, 0.0000000e+00, 1.7634366e-04],
            [1.6495248e+00, 4.0376671e-03, 4.4363579e-03],
            [1.6769145e+00, 8.3101438e-03, 8.7179570e-03],
            [1.6501241e+00, 1.2121308e-02, 1.2956701e-02],
            [1.6484656e+00, 1.6134122e-02, 1.7214257e-02],
            [1.6626340e+00, 2.0471876e-02, 2.1498846e-02],
            [1.6509066e+00, 2.4266895e-02, 2.5736690e-02],
            [1.6537316e+00, 2.8398300e-02, 3.0002527e-02],
            [1.6610764e+00, 3.2709800e-02, 3.4283698e-02],
            [1.6876860e+00, 3.7830908e-02, 3.8651604e-02]], dtype=np.float32)
        ]
    np.testing.assert_allclose(pointings_list, pointing_reference, atol=1e-8)



def test_PointingSys_add_single_offset_to_spacecraft():
    sim, dets = gen_simulation_and_dets()

    pointing_sys = lbs.PointingSys(sim.instrument, dets)
    single_offset = np.deg2rad(1.0)
    axis = 'x'
    pointing_sys.spacecraft.add_offset(single_offset, axis)

    sim.create_observations(detectors=dets)


    lbs.prepare_pointings(
        sim.observations,
        sim.instrument,
        sim.spin2ecliptic_quats,
        hwp=sim.hwp
        )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(det_idx, pointings_dtype=np.float32)
            pointings_list.append(pointings)

    pointing_reference = [
        np.array([
            [1.67551601e+00, 3.48855772e-18, 1.62564003e+00],
            [1.67550623e+00, 4.14894568e-03, 1.62991035e+00],
            [1.67547679e+00, 8.29779729e-03, 1.63418043e+00],
            [1.67542768e+00, 1.24464575e-02, 1.63845026e+00],
            [1.67535901e+00, 1.65948346e-02, 1.64271998e+00],
            [1.67527068e+00, 2.07428336e-02, 1.64698911e+00],
            [1.67516267e+00, 2.48903558e-02, 1.65125787e+00],
            [1.67503512e+00, 2.90373117e-02, 1.65552604e+00],
            [1.67488790e+00, 3.31836045e-02, 1.65979338e+00],
            [1.67472112e+00, 3.73291373e-02, 1.66406012e+00]], dtype=np.float32),
        np.array([
            [1.67551601e+00, 0.00000000e+00, 1.76343659e-04],
            [1.67550623e+00, 4.14894568e-03, 4.44655120e-03],
            [1.67547679e+00, 8.29779729e-03, 8.71666055e-03],
            [1.67542768e+00, 1.24464575e-02, 1.29865687e-02],
            [1.67535901e+00, 1.65948346e-02, 1.72561817e-02],
            [1.67527068e+00, 2.07428336e-02, 2.15253998e-02],
            [1.67516267e+00, 2.48903558e-02, 2.57941186e-02],
            [1.67503512e+00, 2.90373117e-02, 3.00622415e-02],
            [1.67488790e+00, 3.31836045e-02, 3.43296826e-02],
            [1.67472112e+00, 3.73291373e-02, 3.85963246e-02]], dtype=np.float32)
        ]
    np.testing.assert_allclose(pointings_list, pointing_reference)

def test_PointingSys_add_common_disturb_to_spacecraft():
    sim, dets = gen_simulation_and_dets()

    nquats = sim.spin2ecliptic_quats.quats.shape[0]
    noise_rad_1d_array = np.zeros(nquats)

    sim.init_random(random_seed=12_345)
    lbs.add_white_noise(noise_rad_1d_array, sigma=np.deg2rad(1), random=sim.random)

    pointing_sys = lbs.PointingSys(sim.instrument, dets)
    axis = 'x'
    pointing_sys.spacecraft.add_disturb(start_time, sampling_hz, noise_rad_1d_array, axis)

    sim.create_observations(detectors=dets)


    lbs.prepare_pointings(
        sim.observations,
        sim.instrument,
        sim.spin2ecliptic_quats,
        hwp=sim.hwp
        )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(det_idx, pointings_dtype=np.float32)
            pointings_list.append(pointings)

    pointing_reference = [
        np.array([
            [1.6892608 , 0.        , 1.62564   ],
            [1.6495248 , 0.00403767, 1.6299001 ],
            [1.6769145 , 0.00831014, 1.6341817 ],
            [1.6501241 , 0.01212131, 1.6384205 ],
            [1.6484656 , 0.01613412, 1.642678  ],
            [1.662634  , 0.02047188, 1.6469626 ],
            [1.6509066 , 0.02426689, 1.6512004 ],
            [1.6537316 , 0.0283983 , 1.6554663 ],
            [1.6610764 , 0.0327098 , 1.6597475 ],
            [1.687686  , 0.03783091, 1.6641153 ]], dtype=np.float32),
        np.array([
            [1.6892608e+00, 0.0000000e+00, 1.7634366e-04],
            [1.6495248e+00, 4.0376671e-03, 4.4363579e-03],
            [1.6769145e+00, 8.3101438e-03, 8.7179570e-03],
            [1.6501241e+00, 1.2121308e-02, 1.2956701e-02],
            [1.6484656e+00, 1.6134122e-02, 1.7214257e-02],
            [1.6626340e+00, 2.0471876e-02, 2.1498846e-02],
            [1.6509066e+00, 2.4266895e-02, 2.5736690e-02],
            [1.6537316e+00, 2.8398300e-02, 3.0002527e-02],
            [1.6610764e+00, 3.2709800e-02, 3.4283698e-02],
            [1.6876860e+00, 3.7830908e-02, 3.8651604e-02]], dtype=np.float32)
        ]

    np.testing.assert_allclose(pointings_list, pointing_reference, atol=1e-8)
