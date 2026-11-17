from threading import Event, Thread
from time import perf_counter

import dask.array as da
import psutil
from dask.callbacks import Callback


def observe(call):
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = Event()
    tasks = 0

    def sample():
        while not stop.wait(0.02):
            samples.append(process.memory_info().rss)

    def count(*args):
        nonlocal tasks
        tasks += 1

    monitor = Thread(target=sample)
    monitor.start()
    try:
        with Callback(posttask=count):
            start = perf_counter()
            output = call()
            call_seconds = perf_counter() - start
            during_call = tasks
            backed = isinstance(output.data, da.Array)
            graph_tasks = len(output.data.__dask_graph__()) if backed else 0
            result = output.compute()
    finally:
        stop.set()
        monitor.join()
        samples.append(process.memory_info().rss)

    peak = max(samples)
    return result, {
        "instrumented_call_seconds": call_seconds,
        "dask_output": backed,
        "lazy_call": backed and during_call == 0,
        "tasks_during_call": during_call,
        "graph_tasks": graph_tasks,
        "executed_tasks": tasks,
        "baseline_rss_bytes": baseline,
        "peak_sampled_rss_bytes": peak,
        "peak_rss_increase_bytes": peak - baseline,
    }
