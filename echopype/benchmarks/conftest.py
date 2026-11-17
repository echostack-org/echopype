import os
import sys
from pathlib import Path

import dask
import pytest
import yaml

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

    with dask.config.set(
        scheduler=config["dask"]["scheduler"],
        num_workers=config["dask"]["num_workers"],
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
