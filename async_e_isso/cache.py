import threading
from typing import Final

THREADS_NUM: Final = 10

cache = {}
history = []


def execute_in_threads() -> None:
    threads = []
    for thread_id in range(THREADS_NUM):
        thread = threading.Thread(target=process_data, args=(thread_id,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert len(history) == THREADS_NUM


def process_data(thread_id: int) -> None:
    for i in range(1_000):
        key = str(i % 10)
        if key not in cache:
            history.append(key)
            cache[key] = f'{thread_id}'
