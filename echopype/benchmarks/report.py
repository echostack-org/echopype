import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from data import open_data


def load(path):
    return {item["fullname"]: item for item in json.loads(path.read_text())["benchmarks"]}


def plot(root, base, head, filename, dataset):
    config = yaml.safe_load((root / "config.yaml").read_text())
    channel = config["datasets"][dataset]["channel"]

    with open_data(
        data_dir=Path(os.environ["BENCHMARK_DATA"]),
        dataset=dataset,
        n_files=int(os.getenv("BENCHMARK_N_FILES", "1")),
        chunks={},
    ) as ds:
        ds = ds.isel(
            ping_time=slice(None, None, max(1, (ds.sizes["ping_time"] + 999) // 1000)),
            range_sample=slice(None, None, max(1, (ds.sizes["range_sample"] + 999) // 1000)),
        )
        sv = ds.Sv.sel(channel=channel).transpose("ping_time", "range_sample")
        depth = ds.depth.sel(channel=channel).isel(ping_time=0)
        valid = np.isfinite(depth.values)

        fig, ax = plt.subplots(figsize=(12, 5))
        mesh = ax.pcolormesh(
            sv.ping_time.values,
            depth.values[valid],
            sv.values[:, valid].T,
            shading="auto",
            cmap="viridis",
            vmin=-90,
            vmax=-30,
        )
        ax.plot(base.ping_time, base, "k-", lw=2, label="Base")
        ax.plot(head.ping_time, head, "w--", lw=1.5, label="Head")
        ax.invert_yaxis()
        ax.set(xlabel="Ping time", ylabel="Depth (m)", title=Path(filename).stem)
        ax.legend(loc="upper right")
        fig.colorbar(mesh, ax=ax, label="Sv (dB)")
        fig.savefig(root / Path(filename).with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def compare(root):
    base, head = (load(root / f"{ref}.json") for ref in ("base", "head"))
    assert base and base.keys() == head.keys(), "Missing or unmatched benchmark cases"

    print("| Case | Metric | Base | Head |")
    print("|---|---|---:|---:|")
    errors = []

    for case, b in base.items():
        h = head[case]
        bm, hm = b["extra_info"], h["extra_info"]

        def row(metric, before, after):
            print(f"| {case} | {metric} | {before} | {after} |")

        for key in ("median", "min", "max"):
            row(f"{key} (s)", f"{b['stats'][key]:.4f}", f"{h['stats'][key]:.4f}")

        delta = (h["stats"]["median"] / b["stats"]["median"] - 1) * 100
        row("Median change", "-", f"{delta:+.1f}%")

        for key in bm:
            if key in ("output", "dataset"):
                continue
            if key.endswith("_bytes"):
                row(
                    key.replace("_bytes", " (MiB)"),
                    f"{bm[key] / 2**20:.1f}",
                    f"{hm[key] / 2**20:.1f}",
                )
            else:
                row(key, bm[key], hm[key])

        base_output = xr.load_dataarray(root / "base" / bm["output"])
        head_output = xr.load_dataarray(root / "head" / hm["output"])

        try:
            xr.testing.assert_identical(base_output, head_output)
        except AssertionError as exc:
            errors.append(f"{case}: {exc}")
            row("Output agreement", "reference", "DIFFERENT")
        else:
            row("Output agreement", "reference", "identical")

        plot(root, base_output, head_output, bm["output"], bm["dataset"])

    print("\nMemory sampled separately every 20 ms.")
    print("Task counts use the local scheduler; graph sizes are before optimization.")

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    compare(Path(sys.argv[1]))
