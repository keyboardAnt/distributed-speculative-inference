import pytest

from dsi.types.exception import IncompatibleAppendError
from dsi.types.result import Result, ResultSI


def test_appending_compatible_objects():
    r1 = Result(cost_per_run=[10.0, 20.0])
    r2 = Result(cost_per_run=[30.0])
    r1.extend(r2)
    assert r1.cost_per_run == [10.0, 20.0, 30.0]


def test_type_safety():
    si = ResultSI(cost_per_run=[30.0], num_iters_per_run=[1])
    r = Result(cost_per_run=[10.0, 20.0])
    with pytest.raises(IncompatibleAppendError):
        si.extend(r)


def test_appending_empty_lists():
    r1 = Result(cost_per_run=[10.0, 20.0])
    r2 = Result(cost_per_run=[])
    r1.extend(r2)
    assert r1.cost_per_run == [10.0, 20.0]


def test_appending_to_empty_lists():
    r1 = Result(cost_per_run=[])
    r2 = Result(cost_per_run=[10.0, 20.0])
    r1.extend(r2)
    assert r1.cost_per_run == [10.0, 20.0]


def test_subclass_compatibility():
    si1 = ResultSI(cost_per_run=[10.0], num_iters_per_run=[1])
    si2 = ResultSI(cost_per_run=[20.0], num_iters_per_run=[2])
    si1.extend(si2)
    assert si1.cost_per_run == [10.0, 20.0]
    assert si1.num_iters_per_run == [1, 2]


def test_multiple_appends():
    r = Result(cost_per_run=[10.0])
    r.extend(Result(cost_per_run=[20.0]))
    r.extend(Result(cost_per_run=[30.0]))
    assert r.cost_per_run == [10.0, 20.0, 30.0]
