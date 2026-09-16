"""Case 3: work that waits instead of computing.

The third shape of code the talk needs. ``time.sleep`` releases the GIL while
it waits, exactly like a socket read or a database query does, so the threads
already overlap on ``3.14.7``: the sweep is divided by the number of threads on
*both* interpreters.

That makes this the clearest "removing the GIL changed nothing" case of the
repository: the code was never blocked by the GIL in the first place, so there
is nothing for the free-threaded build to unlock. Together with the other two
cases it gives three outcomes for the same harness:

- ``test_cpu_bound``: shares nothing, GIL-bound => big gain without the GIL.
- ``test_shared_cache`` / ``test_locked_cache``: synchronizes on every
  iteration => no gain, and worse with more threads.
- ``test_io_bound`` (here): already releases the GIL => identical on both.

Read this case on ``walltime`` only: waiting executes almost no instructions,
so under the ``simulation`` instrument it looks ~1000x cheaper than the others
and the sweep there only reflects the cost of the threads themselves.
"""

import time

from scaling import benchmark_scaling, scaling_threads

#: One unit is one 1 ms sleep, so a run is ~200 ms of waiting when sequential.
#: The unit is deliberately short: a single long sleep would not be splittable
#: across threads, while 200 small ones are.
TOTAL_UNITS = 200


def wait(worker_id: int, units: int) -> None:
    """Block ``units`` times without holding the GIL."""
    for _ in range(units):
        time.sleep(0.001)


@scaling_threads
def test_io_bound(benchmark, threads_num: int) -> None:
    benchmark_scaling(
        benchmark,
        wait,
        threads_num=threads_num,
        total_units=TOTAL_UNITS,
    )
