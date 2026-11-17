import xarray as xr
from metrics import observe

from echopype.mask import detect_seafloor

DATASET = "ek80_hake"


def benchmark_bottom_blackwell(benchmark, output_dir, config, dataset_loader):
    dataset = config["datasets"][DATASET]

    with dataset_loader(DATASET) as ds:

        def call():
            return detect_seafloor(
                ds,
                method="blackwell",
                params={
                    "var_name": "Sv",
                    "channel": dataset["channel"],
                    "threshold": (-40, 2.4, 1.0),
                    "offset": 0.5,
                    "r0": 10,
                    "r1": 1000,
                },
            )

        observed, metrics = observe(call)

        result = benchmark.pedantic(
            lambda: call().compute(),
            **config["benchmark"],
        )

        xr.testing.assert_identical(result, observed)
        assert result.dims == ("ping_time",)

        filename = f"bottom_blackwell_{DATASET}.nc"
        result.to_netcdf(output_dir / filename)

        benchmark.extra_info.update(
            metrics,
            output=filename,
            dataset=DATASET,
        )
