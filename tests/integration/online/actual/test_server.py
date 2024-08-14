from multiprocessing import Pipe, Queue
from unittest.mock import Mock, patch

import pytest

from dsi.online.actual.model import Model
from dsi.online.actual.server import GenerationComplete, Server, ServerDrafter
from dsi.online.actual.state import State


@pytest.fixture
def drafter_setup():
    queue = Queue()
    msg_bus = Queue()
    res_receiver, res_sender = Pipe(duplex=False)
    res_sender.send = Mock()
    state = State([1, 2, 3])
    model = Mock(spec=Model, gpu_id=0, state=state)
    drafter = ServerDrafter(model, queue, msg_bus, res_sender)
    return drafter, model, queue, msg_bus, res_sender


def test_draft_tokens(drafter_setup):
    drafter, model, queue, msg_bus, res_sender = drafter_setup
    model.draft = Mock(return_value=[4, 5, 6])
    tokens = drafter._draft()
    assert tokens == [4, 5, 6]
    model.draft.assert_called_once_with(5)


def test_preempt_and_resume(drafter_setup):
    drafter, model, queue, msg_bus, res_sender = drafter_setup
    assert not drafter._is_preempted()
    drafter.preempt()
    assert drafter._is_preempted()
    drafter._resume()
    assert not drafter._is_preempted()


# Since Server is an abstract base class (ABC), we'll define a simple concrete class for
# testing.
class TestServer(Server):
    def run(self):
        pass


@pytest.fixture
def server_setup():
    model = Mock(spec=Model, gpu_id=7)
    model.state = Mock(spec=State)
    queue = Queue()
    msg_bus = Queue()
    result_pipe = Mock()
    server = TestServer(model, queue, msg_bus, result_pipe)
    return server


def test_resume_when_halted(server_setup):
    server_setup.halt()
    server_setup.preempt()
    with pytest.raises(GenerationComplete) as excinfo:
        server_setup._resume()
    assert "Completed generating" in str(excinfo.value)
    assert server_setup._is_preempted()


def test_is_preempted_or_halted_preempted(server_setup):
    server_setup.preempt()
    assert server_setup._is_preempted_or_halted()


def test_is_preempted_or_halted_halted(server_setup):
    server_setup.halt()
    assert server_setup._is_preempted_or_halted()


def test_is_preempted_or_halted_neither(server_setup):
    assert not server_setup._is_preempted_or_halted()


def test_preemption_clears_queue(drafter_setup):
    drafter, model, queue, msg_bus, res_sender = drafter_setup
    queue.put((1, [1, 2, 3]))
    drafter.preempt()
    assert drafter._is_preempted()
    assert queue.empty(), "Queue should be cleared upon preemption"


def test_draft_with_lookahead(drafter_setup):
    drafter, model, queue, msg_bus, res_sender = drafter_setup
    model.state.tok_ids = [1, 2, 3, 4]
    tok_ids_drafted = [5, 6, 7, 8, 9]
    model.draft = Mock(return_value=tok_ids_drafted)
    tokens = drafter._draft()
    model.draft.assert_called_once_with(drafter._lookahead)
    assert tokens == tok_ids_drafted, "Drafted tokens do not match expected values"


def test_halt_sends_timestamp_and_clears_queues(drafter_setup):
    drafter, model, queue, msg_bus, res_sender = drafter_setup
    # Adding additional servers to servers list
    other_server = Mock(spec=ServerDrafter)
    drafter.servers = [drafter, other_server]

    with patch("time.time", return_value=123456789.0):
        with pytest.raises(GenerationComplete) as excinfo:
            drafter.halt()
        res_sender.send.assert_called_with(123456789.0)

    assert queue.empty() and msg_bus.empty(), "Queues should be cleared upon halting"
    other_server.halt.assert_called_once(), "Other servers should be halted"
    assert "Completed generating" in str(
        excinfo.value
    ), "GenerationComplete exception should be raised with correct message"
