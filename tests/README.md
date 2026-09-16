# Benchmarks

Every benchmark here answers a single question:

> when the **same** amount of work is split across N threads, does the
> free-threaded interpreter (`3.14.7t`, branch `without-gil`) finish it faster
> than the regular one (`3.14.7`, branch `with-gil`)?

The answer depends on the nature of the code, which is the point of the talk.
So the measurement is kept identical for every case and only the workload
changes.

## The rules of the harness

`scaling.py` implements them, `benchmark_scaling` is the only entry point:

1. **The total work is constant.** `1-thread` is the sequential baseline and
   `8-threads` splits that very same work in eight. A case whose time goes down
   as threads go up is really running in parallel; a case that stays flat is
   serialized (by the GIL, by a lock, or by contention). Giving each thread a
   full workload instead would mix "more work" with "more parallelism" and the
   numbers could not be read.
2. **The workers start together.** A barrier releases all threads at once, so
   the measured region is the parallel phase and not a staircase of threads
   starting one after the other.
3. **`1-thread` runs inline**, without spawning a thread: that is the honest
   sequential baseline.
4. **Same thread counts everywhere** (`1, 2, 4, 8`), so two cases can be put
   side by side. The macro runner used in CI has 16 real cores, so 8 workers
   stay away from oversubscription.

## The cases

| File | What it does | Expected outcome without the GIL |
| --- | --- | --- |
| `test_cpu_bound_benchmarks.py` | pure arithmetic on locals, shares nothing | **big gain**, close to dividing by the thread count |
| `test_cache_benchmarks.py` | the talk's shared `cache`/`history`, unsynchronized and then with a lock | **no gain**, and slower as threads go up (contention) |
| `test_io_bound_benchmarks.py` | sleeps, i.e. code that already releases the GIL | **no change**, both interpreters scale the same |

The CPU-bound case is the control and should never be removed: if a case does
not speed up, it only proves something when a case measured the same way *does*
speed up on the same machine. Otherwise the flat line could just as well be the
benchmark setup.

Measured locally on 8 cores (walltime, best of the run, `PYTHONMALLOC=mimalloc`
on both) — CI numbers come from the macro runner and will differ, but the shape
is what matters:

| Benchmark | 3.14.7 (GIL) | 3.14.7t (free-threaded) |
| --- | --- | --- |
| `test_cpu_bound[1-thread]` | 55.4 ms | 60.3 ms |
| `test_cpu_bound[8-threads]` | 55.4 ms | **8.3 ms** |
| `test_shared_cache[1-thread]` | 36.7 ms | 45.5 ms |
| `test_shared_cache[8-threads]` | 37.9 ms | **152.8 ms** |
| `test_locked_cache[1-thread]` | 86.0 ms | 95.4 ms |
| `test_locked_cache[8-threads]` | 85.3 ms | **405.1 ms** |
| `test_io_bound[1-thread]` | 214.8 ms | 211.4 ms |
| `test_io_bound[8-threads]` | **27.0 ms** | **27.6 ms** |

Same harness, same thread sweep, three different conclusions: how the code is
written is what decides, not the interpreter. Note also that every
`1-thread` baseline is 0–20% *slower* on `3.14.7t` — that is the price of the
free-threaded build on code that does not parallelize.

## Adding a case

Create `tests/test_<case>_benchmarks.py` with this skeleton — that is the whole
contract, the harness takes care of the rest:

```python
"""Case N: one line saying what nature of code this is."""

from scaling import benchmark_scaling, scaling_threads

#: What one unit is, and why this total.
TOTAL_UNITS = 400


def work(worker_id: int, units: int) -> None:
    """The code under study, doing `units` units of work."""
    for _ in range(units):
        ...


@scaling_threads
def test_my_case(benchmark, threads_num: int) -> None:
    benchmark_scaling(
        benchmark,
        work,
        threads_num=threads_num,
        total_units=TOTAL_UNITS,
        # setup=reset_shared_state,  # optional, runs before each measured run
    )
```

Two things to get right:

- `work` must be **splittable**: calling it once with 100 units has to be
  equivalent to calling it twice with 50. Otherwise the sweep is not comparing
  the same workload.
- `TOTAL_UNITS` must be divisible by `8` (the largest thread count) and big
  enough that one sequential run takes a few tens of milliseconds. Below that,
  thread creation (~1–2 ms for 8 threads) dominates and hides the effect.

If the case shares state, pass `setup=` to reset it before each run; it is part
of the measured region, so keep it cheap.

## Running them

```bash
uv run pytest tests/ --codspeed --codspeed-mode=walltime
```

On the free-threaded interpreter, keeping the allocator identical to the one
used in CI:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ft uv sync --locked --python 3.14.7t
UV_PROJECT_ENVIRONMENT=.venv-ft PYTHONMALLOC=mimalloc \
  uv run --python 3.14.7t pytest tests/ --codspeed --codspeed-mode=walltime
```

## Reading the three instruments

CI measures `simulation`, `walltime` and `memory` on every push:

- **`walltime`** is the instrument that answers the question of the talk: it is
  elapsed time, so it is the only one that can show parallelism.
- **`simulation`** counts executed instructions and serializes the threads, so
  it never shows a speedup. What it does show, very stably, is the *extra work*
  a case pays for being threaded (locking, contention, thread setup).
- **`memory`** shows the allocation cost of the same code.

Both branches must run with `PYTHONMALLOC=mimalloc`: the free-threaded build
rejects the `malloc` allocator, and mixing allocators shows up as a fake
free-threading regression (it is worth up to +115% on the threaded benchmarks
by itself).
