from __future__ import annotations

import queue
import typing as tp


@tp.runtime_checkable
class FutureLike(tp.Protocol):
    def result(self) -> tp.Any: ...
    def add_done_callback(self, fn: tp.Callable[[tp.Any], None]) -> None: ...


class DynamicAsCompleted(tp.Iterator[FutureLike]):
    """
    An as_completed iterator that allows adding new futures dynamically.
    Futures added via `add` will be yielded in the order they complete.

    Example:

        >>> dynac = DynamicAsCompleted(initial_futures)
        >>> for future in dynac:
        >>>     result = future.result()
        >>>     if need_more_work(result):
        >>>         dynac.add(submit_new_work())

        # or using iter_batches
        >>> for batch in dynac.iter_batches(batch_size=5):
        >>>     for future in batch:
        >>>         process(future)

    """

    def __init__(self, futures: tp.Iterable[FutureLike]) -> None:
        self._queue: queue.Queue[FutureLike] = queue.Queue()
        self._pending = 0
        self._finished = 0

        for future in futures:
            self.add(future)

    def __iter__(self) -> DynamicAsCompleted:
        return self

    def __next__(self) -> FutureLike | StopIteration:
        # this means we are done
        if self.pending == 0:
            assert self._queue.empty()
            raise StopIteration

        # else wait for the next future to complete
        future = self._queue.get()
        self._pending -= 1
        self._finished += 1
        return future

    @property
    def pending(self) -> int:
        return self._pending

    @property
    def finished(self) -> int:
        return self._finished

    def __repr__(self) -> str:
        pending, finished = self.pending, self.finished
        return f"<DynamicAsCompleted {pending=} {finished=}>"

    def iter_batches(
        self, batch_size: int, *, flush_partial: bool = True
    ) -> tp.Iterator[list[FutureLike]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        batch: list[FutureLike] = []
        while True:
            try:
                future = next(self)
            except StopIteration:
                break
            batch.append(future)
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch and flush_partial:
            yield batch

    def add(self, future: FutureLike) -> FutureLike:
        if not isinstance(future, FutureLike):
            raise TypeError(
                "add expects a future that implements 'add_done_callback' instance"
            )

        self._pending += 1
        future.add_done_callback(self._queue.put)
        return future


if __name__ == "__main__":
    import concurrent.futures
    import time

    def sleep_and_return(n: int) -> int:
        time.sleep(n)
        return n

    with concurrent.futures.ThreadPoolExecutor() as executor:
        initial_futures = [executor.submit(sleep_and_return, i) for i in [3, 1, 4]]
        dynac = DynamicAsCompleted(initial_futures)

        for future in dynac:
            result = future.result()
            print(f"Completed: {result}")
            if result == 1:
                new_future = executor.submit(sleep_and_return, 2)
                dynac.add(new_future)

        print("All done!")
