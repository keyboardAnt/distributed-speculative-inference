from dsi.types.result import ResultSimul


def test_appending_compatible_objects():
    r1 = ResultSimul(cost_per_repeat=[10.0, 20.0])
    r2 = ResultSimul(cost_per_repeat=[30.0])
    r1.extend(r2)
    assert r1.cost_per_repeat == [10.0, 20.0, 30.0]


def test_type_safety():
    r1 = ResultSimul(cost_per_repeat=[30.0], num_iters_per_repeat=[1])
    r2 = ResultSimul(cost_per_repeat=[10.0, 20.0])
    r1.extend(r2)
    assert r1.cost_per_repeat == [30.0, 10.0, 20.0]
    assert r1.num_iters_per_repeat == [1]
    r2.extend(r1)
    assert r2.cost_per_repeat == [10.0, 20.0, 30.0, 10.0, 20.0]
    assert r2.num_iters_per_repeat == [1]


def test_appending_empty_lists():
    r1 = ResultSimul(cost_per_repeat=[10.0, 20.0])
    r2 = ResultSimul(cost_per_repeat=[])
    r1.extend(r2)
    assert r1.cost_per_repeat == [10.0, 20.0]


def test_appending_to_empty_lists():
    r1 = ResultSimul(cost_per_repeat=[])
    r2 = ResultSimul(cost_per_repeat=[10.0, 20.0])
    r1.extend(r2)
    assert r1.cost_per_repeat == [10.0, 20.0]


def test_subclass_compatibility():
    si1 = ResultSimul(cost_per_repeat=[10.0], num_iters_per_repeat=[1])
    si2 = ResultSimul(cost_per_repeat=[20.0], num_iters_per_repeat=[2])
    si1.extend(si2)
    assert si1.cost_per_repeat == [10.0, 20.0]
    assert si1.num_iters_per_repeat == [1, 2]


def test_multiple_appends():
    r = ResultSimul(cost_per_repeat=[10.0])
    r.extend(ResultSimul(cost_per_repeat=[20.0]))
    r.extend(ResultSimul(cost_per_repeat=[30.0]))
    assert r.cost_per_repeat == [10.0, 20.0, 30.0]
