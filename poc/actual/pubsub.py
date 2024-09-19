from poc.actual.event import Preemption


import asyncio
from typing import Dict


class PubSub:
    def __init__(self):
        self.queue: asyncio.Queue[Preemption] = asyncio.Queue()
        self.subscribers: Dict[int, asyncio.Queue[Preemption]] = {}
        self.ready = asyncio.Event()
        print("PubSub: Initialized with 0 subscribers")

    async def publish(self, message: Preemption) -> None:
        await self.queue.put(message)
        print(
            f"PubSub: Published message '{message}'. Queue size: {self.queue.qsize()}"
        )

    async def subscribe(self, worker_id: int) -> asyncio.Queue[Preemption]:
        if worker_id in self.subscribers:
            print(f"PubSub: GPU {worker_id} already subscribed. Replacing existing queue.")
            # Delete the old queue
            del self.subscribers[worker_id]

        # Create a new queue
        subscriber = asyncio.Queue()
        self.subscribers[worker_id] = subscriber
        print(
            f"PubSub: New subscriber added for GPU {worker_id}."
            f" Total subscribers: {len(self.subscribers)}"
        )
        return subscriber

    async def broadcast(self) -> None:
        print("PubSub: Starting broadcast loop")
        self.ready.set()  # Signal that the broadcast loop is ready
        while True:
            try:
                message = await self.queue.get()
                print(
                    f"PubSub: Broadcasting message '{message}' to"
                    f" {len(self.subscribers)} subscribers"
                )
                for subscriber in self.subscribers.values():
                    await subscriber.put(message)
                print(f"PubSub: Broadcast complete. Queue size: {self.queue.qsize()}")
            except asyncio.CancelledError:
                print("PubSub: Broadcast task was cancelled")
                raise
            except Exception as e:
                print(f"PubSub: Exception in broadcast loop: {e}")