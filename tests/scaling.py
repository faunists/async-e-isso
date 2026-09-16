"""Harness shared by every benchmark case of this repository.

Each case answers the same question:

    when the *same* amount of work is split across N threads, does the
    free-threaded interpreter (``3.14.7t``, branch ``without-gil``) finish it
    faster than the regular one (``3.14.7``, branch ``with-gil``)?

The answer depends on the nature of the code, and that is the whole point of
the talk. To make the comparison readable, every case follows the same rules:

1. **The total amount of work is constant.** ``1-thread`` is the sequential
   baseline and ``8-threads`` splits the very same work in eight. A benchmark
   whose time goes down when threads go up is really running in parallel; one
   that stays flat is serialized (by the GIL, by a lock, or by contention on
   shared state). If every thread did a full workload instead, the numbers
   would mix "more work" with "more parallelism" and could not be read.
2. **The workers start together.** A barrier releases all the threads at the
   same time, so the measured region is the parallel phase instead of a
   staircase of threads starting one after another.
3. **Only the workload changes between cases.** Same harness, same thread
   counts, same total work, so two cases can be compared side by side.

How to read the results on CodSpeed:

- ``walltime`` is the instrument that answers the question above: it measures
  elapsed time, so it is the only one that can show parallelism.
- ``simulation`` counts executed instructions. Threads are serialized under the
  simulator, so it never shows a speedup; what it does show is the *extra work*
  (locking, contention, thread setup) that a case pays for being threaded.
- ``memory`` shows the allocation cost of the same code.

Adding a new case
-----------------

Copy ``tests/test_cache_benchmarks.py`` and change three things:

- ``work(worker_id, units)``: the code under study, doing ``units`` units of
  work. It must be splittable, i.e. calling it once with 100 units has to be
  equivalent to calling it twice with 50.
- ``TOTAL_UNITS``: how many units make up one benchmark run. Pick a number
  divisible by ``max(THREAD_COUNTS)`` and big enough that one run takes a few
  tens of milliseconds, otherwise thread startup (~1-2 ms for 8 threads)
  dominates the measurement and hides the effect.
- ``setup``/assertions: whatever the case needs to start from a clean state.
"""

import threading
from collections.abc import Callable
from typing import Final

import pytest

#: Thread counts every case is measured with. ``1`` is the sequential
#: baseline; the CodSpeed macro runner the workflow uses has 16 real cores, so
#: up to 8 workers stay away from oversubscription.
THREAD_COUNTS: Final = (1, 2, 4, 8)

#: Applies ``THREAD_COUNTS`` to a test, naming the cases ``[1-thread]``,
#: ``[2-threads]``, ... so the sweep is readable in the CodSpeed dashboard.
scaling_threads: Final = pytest.mark.parametrize(
    'threads_num',
    THREAD_COUNTS,
    ids=lambda threads_num: f'{threads_num}-thread{"s" if threads_num > 1 else ""}',
)

#: A piece of work, receiving the id of the worker running it and the number of
#: work units it has to perform.
Work = Callable[[int, int], None]


def run_in_threads(work: Work, threads_num: int, total_units: int) -> None:
    """Run ``total_units`` units of ``work``, split across ``threads_num``.

    The threads wait on a barrier so that they all start working at the same
    time, and ``threads_num=1`` runs the work inline: a plain function call is
    the honest sequential baseline to compare the other cases against.
    """
    units, remainder = divmod(total_units, threads_num)
    if remainder:
        raise ValueError(
            f'{total_units} units cannot be split evenly across {threads_num} threads',
        )

    if threads_num == 1:
        work(0, units)
        return

    barrier = threading.Barrier(threads_num)

    def worker(worker_id: int) -> None:
        barrier.wait()
        work(worker_id, units)

    threads = [
        threading.Thread(target=worker, args=(worker_id,))
        for worker_id in range(threads_num)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def benchmark_scaling(
    benchmark,
    work: Work,
    threads_num: int,
    total_units: int,
    setup: Callable[[], None] | None = None,
) -> None:
    """Benchmark ``work`` spread over ``threads_num`` threads.

    ``setup`` is called before each run to reset whatever state the case
    shares; keep it cheap, it is part of the measured region (as is thread
    creation, which is a real cost of threading the code).
    """

    def run() -> None:
        if setup is not None:
            setup()
        run_in_threads(work, threads_num, total_units)

    benchmark(run)
