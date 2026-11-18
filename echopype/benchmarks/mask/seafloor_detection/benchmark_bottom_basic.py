import xarray as xr
from metrics import observe

from echopype.mask import detect_seafloor

DATASET = "ek80_hake"


def benchmark_bottom_basic(benchmark, output_dir, config, dataset_loader):
    dataset = config["datasets"][DATASET]

    with dataset_loader(DATASET) as ds:

        def call():
            return detect_seafloor(
                ds,
                method="basic",
                params={
                    "var_name": "Sv",
                    "channel": dataset["channel"],
                    "threshold": (-40, -20),
                    "offset_m": 0.3,
                    "bin_skip_from_surface": 200,
                },
            )

        observed, metrics = observe(call)

        result = benchmark.pedantic(
            lambda: call().compute(),
            **config["benchmark"],
        )

        xr.testing.assert_identical(result, observed)

        filename = f"bottom_basic_{DATASET}.nc"
        result.to_netcdf(output_dir / filename)

        benchmark.extra_info.update(
            metrics,
            output=filename,
            dataset=DATASET,
        )
