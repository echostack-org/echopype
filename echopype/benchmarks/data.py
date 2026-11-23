import json
from contextlib import contextmanager

import xarray as xr


@contextmanager
def open_data(data_dir, dataset, n_files, chunks):
    metadata = json.loads((data_dir / f"{dataset}.json").read_text())
    n_pings = metadata["ping_boundaries"][n_files - 1]

    with xr.open_zarr(data_dir / f"{dataset}.zarr", chunks={}) as source:
        ds = source.isel(ping_time=slice(0, n_pings))
        yield ds.load() if chunks is None else ds.chunk(chunks)
