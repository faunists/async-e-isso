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
   `32-threads` splits that very same work in thirty-two. A case whose time goes
   down as threads go up is really running in parallel; a case that stays flat is
   serialized (by the GIL, by a lock, or by contention). Giving each thread a
   full workload instead would mix "more work" with "more parallelism" and the
   numbers could not be read.
2. **The workers start together.** A barrier releases all threads at once, so
   the measured region is the parallel phase and not a staircase of threads
   starting one after the other.
3. **`1-thread` runs inline**, without spawning a thread: that is the honest
   sequential baseline.
4. **Same thread counts everywhere** (`1, 2, 4, 8, 16, 32`), so two cases can be
   put side by side. The macro runner used in CI has 16 real cores with no SMT,
   so the sweep crosses the interesting line on purpose:
   - `2`–`8`: fewer workers than cores, everybody gets a core of its own.
   - `16`: one worker per core, the best case for code that really parallelizes.
   - `32`: twice as many workers as cores. Nothing more can run at the same
     time, so this point only adds coordination — context switches, scheduling,
     and more threads fighting over whatever is shared. It is the answer to
     "what does oversubscription cost this kind of code?", and the free-threaded
     build is where it hurts the most, since under the GIL the threads were
     never running at the same time to begin with.

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
on both) — CI numbers come from the 16-core macro runner and will differ, but
the shape is what matters. On this 8-core machine the `32-threads` column is
already 4x oversubscribed, which is exactly what it is there to show:

| Benchmark | 3.14.7 (GIL) | 3.14.7t (free-threaded) |
| --- | --- | --- |
| `test_cpu_bound[1-thread]` | 54.5 ms | 60.4 ms |
| `test_cpu_bound[8-threads]` | 56.0 ms | **8.8 ms** |
| `test_cpu_bound[32-threads]` | 58.7 ms | 10.7 ms |
| `test_shared_cache[1-thread]` | 36.2 ms | 44.0 ms |
| `test_shared_cache[8-threads]` | 38.0 ms | **152.4 ms** |
| `test_shared_cache[32-threads]` | 41.0 ms | **371.1 ms** |
| `test_locked_cache[1-thread]` | 80.1 ms | 93.8 ms |
| `test_locked_cache[8-threads]` | 82.1 ms | **464.2 ms** |
| `test_locked_cache[32-threads]` | 84.5 ms | **812.6 ms** |
| `test_io_bound[1-thread]` | 270.6 ms | 270.6 ms |
| `test_io_bound[8-threads]` | **34.6 ms** | **35.0 ms** |
| `test_io_bound[32-threads]` | 11.3 ms | 12.2 ms |

Same harness, same thread sweep, three different conclusions: how the code is
written is what decides, not the interpreter. Note also that every
`1-thread` baseline is 0–20% *slower* on `3.14.7t` — that is the price of the
free-threaded build on code that does not parallelize.

The tail of the sweep says something the first half cannot. Past the number of
cores nothing more can run at the same time, so those points only add
coordination:

- **CPU-bound** barely moves (8.8 → 10.7 ms): threads that share nothing just
  get time-sliced, so oversubscription costs a few percent and no more.
- **Shared and locked cache** get much worse (152 → 371 ms, 464 → 813 ms):
  every extra thread is another one fighting for the same cache line or the
  same lock, and a thread descheduled inside the critical section blocks all
  the others. This is the shape to look for when someone says "it is thread
  safe, we just added a lock".
- **I/O-bound** keeps improving (34.6 → 11.3 ms, identically on both
  interpreters): waiting threads do not need a core, so oversubscription is
  the normal way to run that kind of code.
- Under the GIL, all three are flat from end to end — there was never anything
  running in parallel to oversubscribe in the first place.

Expect the `16-threads` and `32-threads` points of the contended cases to be
the noisiest of the sweep (20–40% relative standard deviation locally): once
threads fight over a lock, the scheduler decides the result.

## Adding a case

Create `tests/test_<case>_benchmarks.py` with this skeleton — that is the whole
contract, the harness takes care of the rest:

```python
"""Case N: one line saying what nature of code this is."""

from scaling import benchmark_scaling, scaling_threads

#: What one unit is, and why this total. Must divide by 32.
TOTAL_UNITS = 384


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
- `TOTAL_UNITS` must be divisible by `32` (the largest thread count) and big
  enough that one sequential run takes a few tens of milliseconds. Below that,
  thread creation (~1–2 ms for 8 threads, ~5 ms for 32) dominates and hides the
  effect.

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
