import asyncio
import signal

import aioconsole


class AsyncComputeServer:
    def __init__(self):
        self.task = None
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, self.preempt_computation)
        loop.add_signal_handler(signal.SIGTERM, self.preempt_computation)

    async def long_running_task(self):
        try:
            print("Starting long operation...")
            await asyncio.sleep(
                5
            )  # Simulate a long-running operation like GPU computation
            print("Operation completed successfully.")
        except asyncio.CancelledError:
            print("Operation was preempted!")
        finally:
            self.task = None  # Reset task after completion or cancellation

    def preempt_computation(self):
        if self.task and not self.task.done():
            self.task.cancel()
            print("Preemption signal received. Cancelling the operation...")

    async def handle_requests(self):
        while True:
            command = (
                (
                    await aioconsole.ainput(
                        "Enter 'start' to initiate computation or 'exit' to quit: "
                    )
                )
                .strip()
                .lower()
            )
            if command == "start":
                if self.task is None or self.task.done():
                    self.task = asyncio.create_task(self.long_running_task())
                else:
                    print(
                        "A task is already running. Please preempt it first if you want"
                        " to start a new one."
                    )
            elif command == "exit":
                if self.task:
                    self.task.cancel()
                break


async def main():
    server = AsyncComputeServer()
    await server.handle_requests()


if __name__ == "__main__":
    asyncio.run(main())
