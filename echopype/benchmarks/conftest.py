import os
import sys
from pathlib import Path

import dask
import pytest
import yaml
from data import open_data
from distributed import Client, LocalCluster

sys.path.insert(0, str(Path(__file__).resolve().parent))

if "BENCHMARK_SOURCE" in os.environ:
    sys.path.insert(0, os.environ["BENCHMARK_SOURCE"])

import echopype  # noqa: E402


@pytest.fixture(scope="session")
def config():
    with Path(__file__).with_name("config.yaml").open() as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session", autouse=True)
def scheduler(config):
    if "BENCHMARK_SOURCE" in os.environ:
        assert Path(echopype.__file__).resolve() == (
            Path(os.environ["BENCHMARK_SOURCE"]).resolve() / "echopype/__init__.py"
        )

    settings = config["dask"]
    if settings["scheduler"] == "distributed":
        with (
            LocalCluster(
                n_workers=settings["num_workers"],
                threads_per_worker=settings["threads_per_worker"],
                memory_limit=settings["memory_limit"],
                dashboard_address=None,
            ) as cluster,
            Client(cluster),
        ):
            yield
    else:
        with dask.config.set(
            scheduler=settings["scheduler"],
            num_workers=settings.get("num_workers", 1),
        ):
            yield


@pytest.fixture
def output_dir():
    path = Path(os.getenv("BENCHMARK_OUTPUT", ".cache/benchmarks/outputs"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def data_dir():
    return Path(os.environ["BENCHMARK_DATA"])


@pytest.fixture
def dataset_loader(config, data_dir):
    def load(name):
        return open_data(
            data_dir=data_dir,
            dataset=name,
            n_files=int(os.getenv("BENCHMARK_N_FILES", "1")),
            chunks=config["chunks"],
        )

    return load
