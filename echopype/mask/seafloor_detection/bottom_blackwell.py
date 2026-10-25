import numpy as np
import scipy.ndimage as ndima
import xarray as xr
import dask.array as da
import dask_image.ndfilters as dask_ndfilters

from echopype.mask.seafloor_detection.utils import _check_inputs, _parse_blackwell_thresholds
from echopype.utils.compute import _lin2log, _log2lin


def _dask_nanmedian(data: da.Array) -> float:
    """Compute the exact median of finite values without materializing masked cells."""
    finite_values = data[da.isfinite(data)].compute()
    if finite_values.size == 0:
        return np.nan
    return float(np.median(finite_values))


def _bottom_blackwell_dask(
    Svchunk: xr.DataArray,
    thetachunk: xr.DataArray,
    phichunk: xr.DataArray,
    depth_ref: xr.DataArray,
    range_depth: np.ndarray,
    ping_time: xr.DataArray,
    r0_idx: int,
    r1_idx: int,
    tSv: float,
    ttheta: float,
    tphi: float,
    offset: float,
    wtheta: int,
    wphi: int,
    channel: str,
) -> xr.DataArray:
    """Run Blackwell detection with chunked arrays and Dask-image operations."""
    sv_data = da.asarray(Svchunk.data)
    theta_data = da.asarray(thetachunk.data).rechunk(sv_data.chunks)
    phi_data = da.asarray(phichunk.data).rechunk(sv_data.chunks)

    theta_mask = dask_ndfilters.uniform_filter(
        theta_data, size=wtheta, mode="reflect"
    ) ** 2 > ttheta
    phi_mask = dask_ndfilters.uniform_filter(phi_data, size=wphi, mode="reflect") ** 2 > tphi
    angle_mask = theta_mask | phi_mask

    angle_mask_data = angle_mask.compute()
    if angle_mask_data.any():
        dask_angle_mask = da.from_array(angle_mask_data, chunks=sv_data.chunks)
        selected_sv_linear = da.where(
            dask_angle_mask, _log2lin(sv_data), np.nan
        )
        median_sv = _lin2log(_dask_nanmedian(selected_sv_linear))
        if np.isnan(median_sv):
            median_sv = np.inf
        median_sv = max(float(median_sv), tSv)

        sv_mask_data = (sv_data > median_sv).compute()
        components, component_count = ndima.label(
            sv_mask_data, structure=ndima.generate_binary_structure(2, 2)
        )
        del sv_mask_data, selected_sv_linear, dask_angle_mask

        intercepted = np.unique(components[angle_mask_data])
        intercepted = intercepted[intercepted != 0]
        component_lookup = np.zeros(component_count + 1, dtype=bool)
        component_lookup[intercepted] = True
        mask_chunk = component_lookup[components]
        del components, component_lookup, angle_mask_data
    else:
        mask_chunk = da.zeros(sv_data.shape, chunks=sv_data.chunks, dtype=bool)
        del angle_mask_data

    mask = da.pad(
        mask_chunk,
        ((r0_idx, len(range_depth) - r1_idx), (0, 0)),
        mode="constant",
    )
    bottom_sample_idx = mask.argmax(axis=0)
    depth_lookup = depth_ref.isel(
        range_sample=xr.DataArray(
            bottom_sample_idx,
            dims=["ping_time"],
            coords={"ping_time": ping_time},
        )
    )
    bottom_depth = depth_lookup - offset

    return xr.DataArray(
        bottom_depth.data,
        dims=["ping_time"],
        coords={"ping_time": ping_time},
        name="bottom_depth",
        attrs={
            "detector": "blackwell",
            "threshold_Sv": float(tSv),
            "threshold_angle_major": float(ttheta),
            "threshold_angle_minor": float(tphi),
            "offset_m": float(offset),
            "channel": channel,
        },
    )


def bottom_blackwell(
    ds: xr.Dataset,
    var_name: str,
    channel: str,
    threshold: float | list | tuple = -75,
    offset: float = 0.3,
    r0: float = 0,
    r1: float = 500,
    wtheta: int = 28,
    wphi: int = 52,
) -> xr.DataArray:
    """
     Shoal detector modified from the "blackwell" function
     in `mask_seabed.py`, originally written by
     Alejandro ARIZA for the Echopy library © 2020.

     Based on: "Blackwell et al (2019), Aliased seabed detection in fisheries acoustic
     data". (https://arxiv.org/abs/1904.10736)

     Overview
     ---------
     1) Range crop: restrict processing to r ∈ [r0, r1].

     2) Angle smoothing: convolve along-ship (θ) and athwart-ship (ϕ) angles
         with square kernels of size wtheta and wphi (mean filters).

     3) Angle activity mask: flag pixels where the smoothed angles’ squared values
         exceed thresholds (θ > ttheta or ϕ > tphi), then take the union.
    4) Adaptive Sv threshold: compute the median Sv over the angle mask in linear
         units (then convert to dB), set it to tSv if lower, and threshold Sv > Sv_median.

     5) Connected components: label Sv above threshold patches and keep only those
         intersecting the angle mask.

     6) Bottom pick: for each ping, take the shallowest range of the kept patch and
         subtract an `offset` (m) to place the bottom line slightly above it.

     Parameters
     ----------
     ds : xr.Dataset
         Dataset containing:
           • ``var_name`` (Sv in dB) with dims typically
             (``channel``, ``ping_time``, ``range_sample``),
           • ``angle_alongship`` and ``angle_athwartship`` with compatible dims,
           • a vertical coordinate (e.g., ``depth``) aligned with ``range_sample``.
     var_name : str
         Name of the Sv variable to use (e.g., ``"Sv"``).
     channel : str
         Channel identifier to process (must match an entry in ``ds['channel']``).
     threshold : float | list | tuple, default -75
         Either a single Sv threshold in dB (angle thresholds use defaults), or a
         3-tuple/list ``(tSv_dB, ttheta, tphi)`` where ``ttheta`` and ``tphi`` are
         post-smoothing angle activity thresholds (same units as the squared,
         smoothed angles in your pipeline).
     offset : float, default 0.3
         Meters subtracted from the detected range to place the bottom line slightly
         above the echo maximum.
     r0, r1 : float, default 0, 500
         Shallow and deep bounds (meters) of the search interval.
     wtheta, wphi : int, default 28, 52
         Side length (pixels) of the square smoothing windows for the along-ship
         and athwart-ship angle fields.

     Returns
     -------
     xr.DataArray
         1-D bottom depth per ``ping_time`` with attributes:
         ``detector='blackwell'``, ``threshold_Sv``, ``threshold_angle_major``,
         ``threshold_angle_minor``, ``offset_m``, and ``channel``.

     Notes
     -----
     Dask-backed inputs use chunked angle filtering and thresholding. The
     thresholded search window is then labeled with SciPy to avoid the large
     intermediate graphs and memory overhead of distributed component
     reconciliation. The global median is computed exactly from finite,
     angle-selected Sv values.
    """

    # Validate input variables and structure
    Sv_sel, depth_sel = _check_inputs(
        ds,
        var_name=var_name,
        channel=channel,
        required_vars=["angle_alongship", "angle_athwartship"],
    )

    # parse thresholds
    tSv, ttheta, tphi = _parse_blackwell_thresholds(threshold)

    # Direct selection without using ch_sel
    theta = ds["angle_alongship"].sel(channel=channel)
    phi = ds["angle_athwartship"].sel(channel=channel)

    # to match with blackwell echopy format
    Sv_sel = Sv_sel.transpose("range_sample", "ping_time")
    theta = theta.transpose("range_sample", "ping_time")
    phi = phi.transpose("range_sample", "ping_time")

    ping_time = Sv_sel.coords["ping_time"]

    # Use the ping with the deepest valid depth instead of assuming ping_time=0
    max_depth_per_ping = depth_sel.max(dim="range_sample", skipna=True)
    deepest_ping_idx = int(max_depth_per_ping.argmax(dim="ping_time").values)

    depth_ref = depth_sel.isel(ping_time=deepest_ping_idx, drop=True)
    r = depth_ref.values

    # Define core detection range
    r0_idx = np.nanargmin(abs(r - r0))
    r1_idx = np.nanargmin(abs(r - r1)) + 1

    Svchunk = Sv_sel[r0_idx:r1_idx, :]
    thetachunk = theta[r0_idx:r1_idx, :]
    phichunk = phi[r0_idx:r1_idx, :]

    if any(
        isinstance(array.data, da.Array)
        for array in (Svchunk, thetachunk, phichunk)
    ):
        return _bottom_blackwell_dask(
            Svchunk,
            thetachunk,
            phichunk,
            depth_ref,
            r,
            ping_time,
            r0_idx,
            r1_idx,
            tSv,
            ttheta,
            tphi,
            offset,
            wtheta,
            wphi,
            channel,
        )

    # Angle masks
    thetamaskchunk = (
        ndima.uniform_filter(
            thetachunk,
            size=wtheta,
            mode="reflect",
        )
        ** 2
        > ttheta
    )

    phimaskchunk = (
        ndima.uniform_filter(
            phichunk,
            size=wphi,
            mode="reflect",
        )
        ** 2
        > tphi
    )
    anglemaskchunk = thetamaskchunk | phimaskchunk

    # Apply Blackwell algorithm
    if anglemaskchunk.any():

        Svmedian_anglemasked = float(
            _lin2log(np.nanmedian(_log2lin(Svchunk.values[anglemaskchunk])))
        )

        if np.isnan(Svmedian_anglemasked):
            Svmedian_anglemasked = np.inf
        if Svmedian_anglemasked < tSv:
            Svmedian_anglemasked = tSv

        Svmaskchunk = Svchunk > Svmedian_anglemasked

        # Connected components
        items = ndima.label(Svmaskchunk, ndima.generate_binary_structure(2, 2))[0]
        intercepted = list(set(items[anglemaskchunk]))
        if 0 in intercepted:
            intercepted.remove(0)

        # Combine intercepted items
        maskchunk = np.zeros(Svchunk.shape, dtype=bool)
        for i in intercepted:
            maskchunk |= items == i

        # Add padding
        above = np.zeros((r0_idx, maskchunk.shape[1]), dtype=bool)
        below = np.zeros((len(r) - r1_idx, maskchunk.shape[1]), dtype=bool)

        mask = np.r_[above, maskchunk, below]

    else:
        mask = np.zeros_like(Sv_sel, dtype=bool)

    # Bottom detection from mask - offset
    bottom_sample_idx = mask.argmax(axis=0)
    bottom_depth = r[bottom_sample_idx] - offset

    # Return 1D DataArray with attributes
    return xr.DataArray(
        bottom_depth,
        dims=["ping_time"],
        coords={"ping_time": ping_time},
        name="bottom_depth",
        attrs={
            "detector": "blackwell",
            "threshold_Sv": float(tSv),
            "threshold_angle_major": float(ttheta),
            "threshold_angle_minor": float(tphi),
            "offset_m": float(offset),
            "channel": channel,
        },
    )
