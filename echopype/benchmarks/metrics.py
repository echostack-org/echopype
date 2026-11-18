import os
from contextlib import contextmanager
from threading import Event, Thread
from time import perf_counter

import dask.array as da
import psutil
from dask.callbacks import Callback
from distributed import get_client, get_task_stream


@contextmanager
def task_events(client):
    events = []
    if client is None:
        with Callback(posttask=lambda *args: events.append(1)):
            yield events
    else:
        with get_task_stream(client=client) as stream:
            yield events
        events.extend(stream.data)


def observe(call):
    try:
        client = get_client()
    except ValueError:
        client = None

    driver = psutil.Process()
    workers = (
        []
        if client is None
        else [
            psutil.Process(pid) for pid in set(client.run(os.getpid).values()) if pid != driver.pid
        ]
    )
    processes = [driver, *workers]

    def rss():
        return [p.memory_info().rss for p in processes]

    def cpu():
        return sum(t.user + t.system for t in (p.cpu_times() for p in processes))

    initial = rss()
    baseline = sum(initial)
    samples = [baseline]
    worker_samples = [sum(initial[1:])]
    stop = Event()

    def sample_once():
        values = rss()
        samples.append(sum(values))
        worker_samples.append(sum(values[1:]))

    def sample():
        while not stop.wait(0.02):
            sample_once()

    monitor = Thread(target=sample)
    cpu_start = cpu()
    start = perf_counter()
    monitor.start()
    try:
        with task_events(client) as call_tasks:
            call_start = perf_counter()
            output = call()
            call_seconds = perf_counter() - call_start

        backed = isinstance(output.data, da.Array)
        graph_tasks = len(output.data.__dask_graph__()) if backed else 0

        with task_events(client) as compute_tasks:
            result = output.compute()
    finally:
        stop.set()
        monitor.join()
        sample_once()

    cpu_seconds = cpu() - cpu_start
    elapsed = perf_counter() - start
    peak = max(samples)
    return result, {
        "instrumented_call_seconds": call_seconds,
        "dask_output": backed,
        "lazy_call": backed and not call_tasks,
        "tasks_during_call": len(call_tasks),
        "graph_tasks": graph_tasks,
        "executed_tasks": len(call_tasks) + len(compute_tasks),
        "baseline_rss_bytes": baseline,
        "peak_sampled_rss_bytes": peak,
        "peak_rss_increase_bytes": peak - baseline,
        "worker_peak_rss_bytes": max(worker_samples),
        "cpu_seconds": round(cpu_seconds, 4),
        "mean_cpu_cores": round(cpu_seconds / elapsed, 3),
    }
