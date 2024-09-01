from multiprocessing import Pipe, Queue
from unittest.mock import MagicMock, Mock, patch

import pytest

from dsi.online.actual.model import Model, SetupModel
from dsi.online.actual.server import (
    GenerationComplete,
    Server,
    ServerDrafter,
    SetupServer,
)
from dsi.online.actual.state import State


@pytest.fixture
def drafter_setup():
    queue = Queue()
    msg_bus = Queue()
    res_receiver, res_sender = Pipe(duplex=False)
    res_sender.send = Mock()
    setup_model = SetupModel(gpu_id=0, state=State([1, 2, 3]), _name="test_model")
    model = MagicMock(spec=Model, setup=setup_model)
    drafter = ServerDrafter(
        SetupServer(
            model=model, _job_queue=queue, _msg_bus=msg_bus, _result_pipe=res_sender
        )
    )
    return drafter, model, queue, msg_bus, res_sender


# Since Server is an abstract base class (ABC), we'll define a simple concrete class for
# testing.
class TestServer(Server):
    def run(self):
        pass


@pytest.fixture
def server():
    setup_model = SetupModel(gpu_id=7, state=State([1, 2, 3]), _name="test_model")
    model = Mock(spec=Model, setup=setup_model)
    model.state = Mock(spec=State)
    queue = Queue()
    msg_bus = Queue()
    result_pipe = Mock()
    server = TestServer(
        SetupServer(
            model=model, _job_queue=queue, _msg_bus=msg_bus, _result_pipe=result_pipe
        )
    )
    return server


def test_preemption_clears_queue(drafter_setup):
    drafter, model, queue, msg_bus, res_sender = drafter_setup
    queue.put((1, [1, 2, 3]))
    drafter._preempt()
    assert drafter._is_preempted()
    assert queue.empty(), "Queue should be cleared upon preemption"


def test_halt_sends_timestamp_and_clears_queues(drafter_setup):
    drafter, model, queue, msg_bus, res_sender = drafter_setup
    # Adding additional servers to servers list
    other_server = Mock(spec=ServerDrafter)
    drafter.servers = [drafter, other_server]

    with patch("time.time", return_value=123456789.0):
        with pytest.raises(GenerationComplete):
            drafter._halt()
        res_sender.send.assert_called_with(123456789.0)

    assert queue.empty() and msg_bus.empty(), "Queues should be cleared upon halting"
    # Assert that a GenerationComplete exception is raised
    with pytest.raises(GenerationComplete):
        drafter._halt()
