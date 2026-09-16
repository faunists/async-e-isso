"""Control case: CPU work that shares nothing.

This is the reference every other case is read against. The workload is pure
arithmetic on local variables: no shared object, no lock, nothing to
synchronize. It is the shape of code the free-threaded build is made for, so
the sweep should go from "flat" on ``with-gil`` (the GIL serializes the
threads) to "roughly divided by the number of threads" on ``without-gil``.

Without this control, a case that does not speed up proves nothing: it could
be the contention, or it could be the benchmark setup. Here, whatever
parallelism the machine can give is visible, so a flat sweep somewhere else is
about that code and not about the measurement.

The same file is also the template for a new case: replace ``spin`` with the
code under study and adjust ``TOTAL_UNITS``.
"""

from scaling import benchmark_scaling, scaling_threads

#: One unit is 1000 iterations of the loop below, ~40 µs. 1600 units are worth
#: a few tens of milliseconds of single threaded work, on par with the other
#: cases.
TOTAL_UNITS = 1600


def spin(worker_id: int, units: int) -> None:
    """Burn CPU without touching anything another thread can see."""
    total = 0
    for _ in range(units):
        for index in range(1_000):
            total += index * index
    assert total >= 0


@scaling_threads
def test_cpu_bound(benchmark, threads_num: int) -> None:
    benchmark_scaling(
        benchmark,
        spin,
        threads_num=threads_num,
        total_units=TOTAL_UNITS,
    )
