# -*- encoding: utf-8 -*-

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-grasp-alm",
        action="store_true",
        default=False,
        help="Regenerate reference a_ℓm data for GRASP beams instead of comparing",
    )


@pytest.fixture
def regenerate_grasp_alm(request):
    """If set, re-compute the reference a_ℓm coefficients for GRASP beams"""
    return request.config.getoption("--regenerate-grasp-alm")
