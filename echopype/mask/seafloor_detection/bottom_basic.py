import xarray as xr

from echopype.mask.seafloor_detection.utils import _get_bottom_depth, _validate_threshold


def bottom_basic(
    ds: xr.Dataset,
    var_name: str,
    channel: str,
    threshold: tuple[float, float] = (-40.0, -20.0),
    offset_m: float = 0.5,
    bin_skip_from_surface: int = 200,
) -> xr.DataArray:
    """Detect the seafloor using Sv thresholds.

    For each ping, identify the first range sample within the specified Sv
    interval after skipping shallow samples. Retrieve its corresponding
    depth and subtract the offset.

    Supports NumPy and Dask-backed inputs, preserving lazy computation.
    Returns NaN for pings without a detection.
    """

    Sv_sel = ds[var_name].sel(channel=channel)
    depth_sel = ds["depth"].sel(channel=channel)
    tmin, tmax = _validate_threshold(threshold)

    # validate the search interval
    if not 0 <= bin_skip_from_surface < Sv_sel.sizes["range_sample"]:
        raise ValueError("bin_skip_from_surface must be within the range_sample dimension")

    # ignore shallow range samples
    Sv_sliced = Sv_sel.isel(range_sample=slice(bin_skip_from_surface, None))
    depth_sliced = depth_sel.isel(range_sample=slice(bin_skip_from_surface, None))

    # threshold mask
    cond = (Sv_sliced > tmin) & (Sv_sliced < tmax)

    # retrieve bottom depth
    bottom_depth = _get_bottom_depth(depth_sliced, cond, offset_m)

    return xr.DataArray(
        bottom_depth.data,
        dims=["ping_time"],
        coords={"ping_time": ds["ping_time"]},
        name="bottom_depth",
        attrs={
            "detector": "basic",
            "threshold_min": float(tmin),
            "threshold_max": float(tmax),
            "offset_m": float(offset_m),
            "bin_skip_from_surface": int(bin_skip_from_surface),
            "channel": str(channel),
        },
    )
