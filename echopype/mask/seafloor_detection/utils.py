import numpy as np
import xarray as xr


def _check_inputs(ds, var_name, channel, required_vars=None):
    """Validate the dataset variables and channel required for detection."""
    if var_name not in ds:
        raise KeyError(f"{var_name!r} not found in dataset")

    if "depth" not in ds:
        raise KeyError(
            "'depth' variable not found in dataset. Use echopype.consolidate.add_depth() first."
        )

    if "channel" not in ds.coords:
        raise ValueError("Dataset must have 'channel' coordinate")

    if channel not in ds.channel:
        raise ValueError(f"Channel {channel!r} not found in dataset")

    for var in required_vars or []:
        if var not in ds:
            raise KeyError(f"Required variable {var!r} not found in dataset")


def _get_bottom_depth(depth, mask, offset_m):
    """Return the first detected depth per ping, or NaN."""
    idx = mask.argmax(dim="range_sample")

    sample = xr.DataArray(
        np.arange(depth.sizes["range_sample"]),
        dims="range_sample",
        coords={"range_sample": depth["range_sample"]},
    )

    bottom_depth = depth.where(sample == idx).max(dim="range_sample")

    return bottom_depth.where(mask.any(dim="range_sample")) - offset_m


def _validate_threshold(threshold):
    """Validate the lower and upper Sv thresholds (tmin, tmax)."""
    tmin, tmax = map(float, threshold)

    if not tmin < tmax:
        raise ValueError("threshold upper bound must be > lower bound")

    return tmin, tmax
