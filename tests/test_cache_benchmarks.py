"""Case 1: a cache shared by every thread.

``async_e_isso.cache.process_data`` reads and writes two module level objects
(``cache`` and ``history``) from every thread, without any synchronization.
Three variants of the same work are measured here:

- ``test_shared_cache``: the code exactly as it is written in the talk. It is
  fast, it scales, and it is *wrong*: the check and the write around ``cache``
  are not atomic, so without the GIL two threads can both decide the key is
  missing and ``history`` ends up longer than the number of threads. See
  ``test_shared_cache_can_lose_the_race``.
- ``test_locked_cache``: the same work, made correct with a lock. This is the
  interesting one for the talk: the critical section is now serialized, so the
  free-threaded interpreter has nothing left to parallelize and the added
  synchronization costs time on both interpreters.
- ``test_execute_in_threads``: the talk's demo function as-is, kept as a
  reference point.

Compare this file with ``tests/test_cpu_bound_benchmarks.py``, which runs the
same sweep on code that shares nothing.
"""

import threading

from async_e_isso import cache as cache_module
from scaling import benchmark_scaling, run_in_threads, scaling_threads

#: One unit of work is one ``process_data`` call, i.e. 1000 lookups in the
#: shared cache. 384 units take a few tens of milliseconds, which keeps thread
#: creation (~1-2 ms for 8 threads, ~5 ms for 32) a small part of the
#: measurement, and they split evenly across every thread count of the sweep.
TOTAL_UNITS = 384

_cache_lock = threading.Lock()


def reset_shared_state() -> None:
    cache_module.cache.clear()
    cache_module.history.clear()


def process_data(worker_id: int, units: int) -> None:
    """``units`` calls to the unsynchronized function from the talk."""
    for _ in range(units):
        cache_module.process_data(worker_id)


def process_data_with_lock(worker_id: int, units: int) -> None:
    """Same work as ``process_data``, with the shared section locked.

    This is the fix for the race condition, and the reason it is here is that
    it shows the cost of that fix: the lock is taken on every iteration, so the
    threads can only ever run one at a time inside the critical section.
    """
    for _ in range(units):
        for index in range(1_000):
            key = str(index % 10)
            with _cache_lock:
                if key not in cache_module.cache:
                    cache_module.history.append(key)
                    cache_module.cache[key] = f'{worker_id}'


@scaling_threads
def test_shared_cache(benchmark, threads_num: int) -> None:
    benchmark_scaling(
        benchmark,
        process_data,
        threads_num=threads_num,
        total_units=TOTAL_UNITS,
        setup=reset_shared_state,
    )

    assert set(cache_module.cache) == {str(index) for index in range(10)}


@scaling_threads
def test_locked_cache(benchmark, threads_num: int) -> None:
    benchmark_scaling(
        benchmark,
        process_data_with_lock,
        threads_num=threads_num,
        total_units=TOTAL_UNITS,
        setup=reset_shared_state,
    )

    assert set(cache_module.cache) == {str(index) for index in range(10)}
    # The lock makes the check-then-write atomic, so no key is ever written
    # twice, on either interpreter.
    assert len(cache_module.history) == 10


def test_execute_in_threads(benchmark) -> None:
    """The demo function of the talk, measured as it is written.

    Every thread runs a full workload here (the work is not split), so this
    benchmark answers a different question than the sweeps above: "how long
    does this exact demo take on each interpreter?".
    """

    def run() -> None:
        reset_shared_state()
        try:
            cache_module.execute_in_threads()
        except AssertionError:
            # `execute_in_threads` asserts `len(history) == THREADS_NUM`. That
            # assertion can legitimately fail on the free-threaded build; it is
            # the race the talk is about, not a benchmark failure.
            pass

    benchmark(run)

    assert set(cache_module.cache) == {str(index) for index in range(10)}


def test_shared_cache_can_lose_the_race() -> None:
    """Documents the race condition. Not a benchmark: CodSpeed skips it.

    On ``3.14.7`` the GIL makes the check-then-write of ``process_data`` look
    atomic and ``history`` always ends up with 10 keys. On ``3.14.7t`` it can
    grow past that, which is why this only asserts the invariant that holds on
    both: a key is never *lost*.
    """
    reset_shared_state()

    run_in_threads(process_data, threads_num=8, total_units=8)

    assert set(cache_module.cache) == {str(index) for index in range(10)}
    assert len(cache_module.history) >= 10
