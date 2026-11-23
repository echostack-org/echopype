import csv
import json
import sys
from pathlib import Path

import numpy as np
import pooch
import xarray as xr
import yaml

import echopype as ep


def get_files(params, config_path):
    if "manifest" in params:
        manifest = config_path.parent / params["manifest"]

        with manifest.open() as f:
            return list(csv.DictReader(f))

    return [{"url": params["url"], "sha256": params["sha256"]}]


def prepare(output):
    config_path = Path(__file__).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())

    output.mkdir(parents=True, exist_ok=True)

    n_files = max(config["scaling"]["file_counts"])
    chunks = config["storage_chunks"]

    for name, params in config["datasets"].items():
        files = get_files(params, config_path)

        if len(files) < n_files:
            raise ValueError(f"{name}: fewer than {n_files} RAW files available")

        sv_paths = []
        manifest = []

        for i, file in enumerate(files[:n_files]):
            print(f"Preparing {i + 1}/{n_files}: {file['url']}", flush=True)

            raw = pooch.retrieve(
                url=file["url"],
                known_hash=file.get("sha256") or None,
                path=Path(pooch.os_cache("echopype")) / "benchmarks",
            )

            ed = ep.open_raw(
                raw,
                sonar_model=params["sonar_model"],
            )

            kwargs = {
                "waveform_mode": params["waveform_mode"],
                "encode_mode": params["encode_mode"],
            }

            ds = ep.calibrate.compute_Sv(ed, **kwargs)

            ds = ep.consolidate.add_depth(
                ds,
                ed,
                depth_offset=params["depth_offset"],
            )

            ds = ep.consolidate.add_splitbeam_angle(
                ds,
                ed,
                **kwargs,
                to_disk=False,
            )

            ds = ds[["Sv", "depth", "angle_alongship", "angle_athwartship"]]
            ds = ds.sel(channel=[params["channel"]])

            index = ds.get_index("ping_time")

            if not index.is_monotonic_increasing or not index.is_unique:
                raise ValueError(f"Invalid ping_time coordinate: {raw}")

            path = output / f"{name}.zarr" if n_files == 1 else output / f"{name}_{i:03d}.zarr"

            ds = ds.drop_encoding().chunk(chunks)
            ds.to_zarr(path, mode="w", consolidated=True)

            sv_paths.append(path)

            manifest.append(
                {
                    "url": file["url"],
                    "sha256": pooch.file_hash(raw),
                    "n_pings": ds.sizes["ping_time"],
                }
            )

            ds.close()
            del ds, ed

        if n_files > 1:
            with xr.open_mfdataset(
                sv_paths,
                engine="zarr",
                combine="nested",
                concat_dim="ping_time",
                data_vars="minimal",
                coords="minimal",
                join="exact",
                compat="equals",
            ) as combined:
                index = combined.get_index("ping_time")

                if not index.is_monotonic_increasing or not index.is_unique:
                    raise ValueError("Invalid combined ping_time coordinate")

                combined = combined.drop_encoding().chunk(chunks)

                combined.to_zarr(
                    output / f"{name}.zarr",
                    mode="w",
                    consolidated=True,
                )

        metadata = {
            "files": manifest,
            "ping_boundaries": np.cumsum([file["n_pings"] for file in manifest]).tolist(),
        }

        (output / f"{name}.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
