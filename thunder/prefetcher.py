from typing import Iterator, TypeVar, Generic
from queue import Queue
from threading import Thread, Lock
import itertools

T_co = TypeVar("T_co", covariant=True)


class Prefetcher(Generic[T_co]):
    def __init__(self, dataloader, prefetch_count: int = 0, num_workers: int = 1):
        self.dataloader = dataloader
        self.prefetch_count = prefetch_count
        self.num_workers = num_workers
        self.queue = Queue(maxsize=prefetch_count)
        self.sentinel = object()
        self.lock = Lock()
        self.workers_done = 0

    def _worker(self):
        with self.lock:
            for item in self.dataloader:
                self.queue.put(item)
        with self.lock:
            self.workers_done += 1
            if self.workers_done == self.num_workers:
                self.queue.put(self.sentinel)

    def __iter__(self) -> Iterator[T_co]:
        if self.prefetch_count > 0:
            # Start the prefetching thread
            self._start_threads()

            # Yield items from the queue
            for item in iter(self.queue.get, self.sentinel):
                yield item
        else:
            # If prefetch_count is 0, just iterate over the dataloader lazily
            with self.lock:
                for item in self.dataloader:
                    yield item

    def _start_threads(self):
        for _ in range(self.num_workers):
            Thread(target=self._worker, daemon=True).start()
