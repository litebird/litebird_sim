# -*- encoding: utf-8 -*-

import litebird_sim as lbs
from time import sleep


def test_time_profiler():
    with lbs.TimeProfiler(name="test", par1=1, par2=2) as profiler:
        sleep(1.1)

    assert profiler.elapsed_time_s() > 1.0
    assert profiler.name == "test"
    assert profiler.valid()
    assert profiler.parameters["par1"] == 1
    assert profiler.parameters["par2"] == 2
