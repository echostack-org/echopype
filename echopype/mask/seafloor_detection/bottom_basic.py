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
    """
    Simple threshold-based seafloor detection returning a 1-D bottom line (depth).

    Summary
    -------
    For the selected `channel`, the algorithm skips the top `bin_skip_from_surface`
    range bins, then finds (per ping) the first range sample where `Sv` is within a
    user-defined dB interval. That depth (looked up from `depth`) minus `offset_m`
    is returned as the bottom.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
          • `var_name` (Sv in dB), typically with dims
            (`channel`, `ping_time`, `range_sample`);
          • `depth` in meters, aligned with `ping_time` and `range_sample`.
    var_name : str
        Name of the Sv variable to use (e.g., `"Sv"`).
    channel : str
        Channel identifier to process (must match an entry in `ds['channel']`).
    threshold : tuple of float, default (-40.0, -20.0)
        Lower and upper Sv thresholds (tmin, tmax) in dB.
    offset_m : float, default 0.5
        Meters subtracted from the detected crossing to place the bottom slightly
        above the detected threshold crossing.
    bin_skip_from_surface : int, default 200
        Number of shallow range bins to ignore before searching (index units,
        not meters).

    Returns
    -------
    xr.DataArray
        1-D bottom depth per `ping_time` (no `channel` dimension) with attributes:
        `detector='basic'`, `threshold_min`, `threshold_max`, `offset_m`,
        `bin_skip_from_surface`, and `channel`.

    Notes
    -----
    * Depth is retrieved from the corresponding depth profile of each ping.
    * Supports NumPy and Dask-backed arrays, preserving lazy execution
        for Dask-backed inputs.
    * If no sample meets the threshold in a ping, NaN is returned.
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
