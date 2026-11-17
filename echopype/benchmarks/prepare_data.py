import sys
from pathlib import Path

import pooch
import yaml

import echopype as ep


def prepare(output):
    config = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    output.mkdir(parents=True, exist_ok=True)

    for name, params in config["datasets"].items():
        raw = pooch.retrieve(
            url=params["url"],
            known_hash=params["sha256"],
            path=Path(pooch.os_cache("echopype")) / "benchmarks",
            fname=params["url"].split("/")[-1],
        )

        ed = ep.open_raw(raw, sonar_model=params["sonar_model"])
        kwargs = {
            "waveform_mode": params["waveform_mode"],
            "encode_mode": params["encode_mode"],
        }

        ds = ep.calibrate.compute_Sv(ed, **kwargs)
        ds = ep.consolidate.add_depth(ds, ed, depth_offset=params["depth_offset"])
        ds = ep.consolidate.add_splitbeam_angle(ds, ed, **kwargs, to_disk=False)

        ds = ds[["Sv", "depth", "angle_alongship", "angle_athwartship"]]
        ds = ds.sel(channel=[params["channel"]])
        ds = ds.isel(ping_time=slice(0, params["n_pings"]))
        ds.to_netcdf(output / f"{name}.nc")


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
