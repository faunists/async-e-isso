import threading
from collections.abc import Callable

import pytest

from async_e_isso import cache as cache_module


def reset_shared_state() -> None:
    cache_module.cache.clear()
    cache_module.history.clear()


def run_in_threads(target: Callable[[int], None], threads_num: int) -> None:
    threads = [
        threading.Thread(target=target, args=(thread_id,))
        for thread_id in range(threads_num)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_process_data_single_thread(benchmark) -> None:
    def run() -> None:
        reset_shared_state()
        cache_module.process_data(0)

    benchmark(run)

    assert cache_module.cache == {str(i): '0' for i in range(10)}


def test_process_data_warm_cache(benchmark) -> None:
    reset_shared_state()
    cache_module.process_data(0)

    benchmark(cache_module.process_data, 0)

    assert len(cache_module.history) == 10


@pytest.mark.parametrize('threads_num', [1, 2, 5, 10])
def test_process_data_in_threads(benchmark, threads_num: int) -> None:
    def run() -> None:
        reset_shared_state()
        run_in_threads(cache_module.process_data, threads_num)

    benchmark(run)

    assert set(cache_module.cache) == {str(i) for i in range(10)}


def test_execute_in_threads(benchmark) -> None:
    def run() -> None:
        reset_shared_state()
        try:
            cache_module.execute_in_threads()
        except AssertionError:
            # This is the race condition the talk is about: without the GIL the
            # `history` list can grow past THREADS_NUM. It is an expected
            # outcome of the demo, not a benchmark failure.
            pass

    benchmark(run)

    assert set(cache_module.cache) == {str(i) for i in range(10)}
