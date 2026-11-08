"""The Simulation.add_* wrappers pass their parameters to the functions
they wrap"""

import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim import simulations


def _simulation(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path, start_time=0.0, duration_s=10.0, random_seed=1
    )
    sim.create_observations(
        detectors=[lbs.DetectorInfo(name="d", sampling_rate_hz=1.0, net_ukrts=1.0)]
    )
    return sim


def _record(monkeypatch, name):
    """Replace simulations.<name> with a function that records its kwargs"""
    calls = []
    monkeypatch.setattr(simulations, name, lambda **kwargs: calls.append(kwargs))
    return calls


def test_add_dipole_passes_the_convolution(tmp_path, monkeypatch):
    sim = _simulation(tmp_path)
    sim.pos_and_vel = object()  # Not used by the recorder
    calls = _record(monkeypatch, "add_dipole_to_observations")
    beam = object()
    sim.add_dipole(
        dipole_type=lbs.DipoleType.LINEAR,
        apply_convolution=True,
        beam_alms=beam,
        frequency_ghz=np.array([140.0]),
        append_to_report=False,
    )
    (kwargs,) = calls
    assert kwargs["apply_convolution"] is True
    assert kwargs["beam_alms"] is beam
    np.testing.assert_array_equal(kwargs["frequency_ghz"], [140.0])


def test_fill_tods_passes_the_harmonics_parameters(tmp_path, monkeypatch):
    sim = _simulation(tmp_path)
    calls = _record(monkeypatch, "scan_map_in_observations")
    phases = {"2f": np.zeros((3, 3))}
    sim.fill_tods(
        maps=object(),
        save_tod=False,
        apply_non_linearity=True,
        add_2f_hwpss=True,
        mueller_phases=phases,
        append_to_report=False,
    )
    (kwargs,) = calls
    assert kwargs["save_tod"] is False
    assert kwargs["apply_non_linearity"] is True
    assert kwargs["add_2f_hwpss"] is True
    assert kwargs["mueller_phases"] is phases


def test_add_noise_passes_the_scale(tmp_path, monkeypatch):
    sim = _simulation(tmp_path)
    calls = _record(monkeypatch, "add_noise_to_observations")
    sim.add_noise(noise_type="white", scale=2.5, append_to_report=False)
    (kwargs,) = calls
    assert kwargs["scale"] == 2.5


def test_unused_parameters_are_deprecated(tmp_path, monkeypatch):
    sim = _simulation(tmp_path)
    _record(monkeypatch, "scan_map_in_observations")
    _record(monkeypatch, "add_convolved_sky_to_observations")
    with pytest.deprecated_call(match="band_filenames"):
        sim.fill_tods(maps=object(), band_filenames=["x"], append_to_report=False)
    with pytest.deprecated_call(match="input_sky_alms_in_galactic"):
        sim.convolve_sky(
            sky_alms=object(),
            beam_alms=object(),
            input_sky_alms_in_galactic=False,
            append_to_report=False,
        )
