import time
from multiprocessing import Process

import pytest

from dsi.online.actual.state import InvalidRollbackError, State


# Define a fixture for initial states
@pytest.fixture
def initial_state():
    return State([1, 2, 3, 4, 5])


# Test initial state properties
def test_initial_state_properties(initial_state):
    assert initial_state.tok_ids == [1, 2, 3, 4, 5]
    assert initial_state.v == 4


# Test setting and getting tok_ids
def test_set_get_tok_ids(initial_state):
    initial_state.tok_ids = [10, 20, 30]
    assert initial_state.tok_ids == [10, 20, 30]


# Test setting and getting v
def test_set_get_v(initial_state):
    initial_state.v = 2
    assert initial_state.v == 2


# Test extend method with verified extension
def test_extend_verified(initial_state):
    initial_state.extend([6, 7], verified=True)
    assert initial_state.tok_ids == [1, 2, 3, 4, 5, 6, 7]
    assert initial_state.v == 6


# Test extend method with unverified extension
def test_extend_unverified(initial_state):
    initial_state.extend([6, 7], verified=False)
    assert initial_state.tok_ids == [1, 2, 3, 4, 5, 6, 7]
    assert initial_state.v == 4  # Should remain unchanged


# Test rollback functionality
def test_rollback(initial_state):
    initial_state.rollback(2)
    assert initial_state.tok_ids == [1, 2, 3]
    assert initial_state.v == 2


# Test rollback with invalid index
def test_rollback_invalid_index(initial_state):
    with pytest.raises(InvalidRollbackError):
        initial_state.rollback(10)


# Test is_aligned method
def test_is_aligned(initial_state):
    from dsi.online.actual.message import MsgVerifiedRightmost

    msg = MsgVerifiedRightmost(v=4, tok_id=5)
    assert initial_state.is_aligned(msg)


# Test clone method
def test_clone(initial_state):
    clone = initial_state.clone(only_verified=True)
    assert clone.tok_ids == [1, 2, 3, 4, 5]
    assert clone.v == 4


def test_multiprocessing_safety():
    state = State([1, 2, 3, 4, 5])

    def modify_state(new_values, delay):
        time.sleep(delay)  # Stagger the start times slightly
        state.tok_ids = new_values
        state.v = len(new_values) - 1

    # Staggered start times to simulate race condition
    delays = [i * 0.01 for i in range(10)]
    new_values_list = [[10 + i, 20 + i, 30 + i] for i in range(10)]
    processes = [
        Process(target=modify_state, args=(new_values, delays[i]))
        for i, new_values in enumerate(new_values_list)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # Check if the last process to modify the state did so correctly
    last_values = new_values_list[-1]
    assert (
        state.tok_ids == last_values
    ), f"Expected {last_values}, but got {state.tok_ids}"
    assert (
        state.v == len(last_values) - 1
    ), f"Expected v to be {len(last_values) - 1}, but got {state.v}"
