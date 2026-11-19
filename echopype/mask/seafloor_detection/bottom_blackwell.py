import dask.array as da
import dask_image.ndfilters as dask_ndfilters
import numpy as np
import scipy.ndimage as ndima
import xarray as xr

from echopype.utils.compute import _lin2log, _log2lin


def bottom_blackwell(
    ds: xr.Dataset,
    var_name: str,
    channel: str,
    threshold: tuple[float, float, float] = (-75.0, 702.0, 282.0),
    offset_m: float = 0.3,
    r0: float = 0,
    r1: float = 500,
    wtheta: int = 28,
    wphi: int = 52,
) -> xr.DataArray:
    """Detect the seafloor using split-beam angles and Sv thresholding.

    Identify connected regions of Sv above an adaptive threshold that
    intersect the smoothed angle mask. Retrieve the first detected depth
    per ping and subtract the offset.

    Adapted from the ``blackwell`` function in ``mask_seabed.py``,
    originally written by Alejandro Ariza for the Echopy library © 2020.
    Based on Blackwell et al. (2019): https://arxiv.org/abs/1904.10736

    Supports NumPy and Dask-backed inputs, with eager computation for
    search-window selection, adaptive thresholding and connected-component
    labeling. Returns NaN for pings without a detection.
    """
    tSv, ttheta, tphi = map(float, threshold)

    # Select channel
    Sv = ds[var_name].sel(channel=channel).transpose("range_sample", "ping_time")
    depth = ds["depth"].sel(channel=channel).transpose("range_sample", "ping_time")
    theta = ds["angle_alongship"].sel(channel=channel).transpose("range_sample", "ping_time")
    phi = ds["angle_athwartship"].sel(channel=channel).transpose("range_sample", "ping_time")

    # Determine search region
    max_depth = depth.max(dim="range_sample", skipna=True)
    deepest_ping = int(max_depth.argmax(dim="ping_time").compute().item())
    depth_ref = depth.isel(ping_time=deepest_ping).values

    r0_idx = np.nanargmin(abs(depth_ref - r0))
    r1_idx = np.nanargmin(abs(depth_ref - r1)) + 1

    search = slice(r0_idx, r1_idx)

    Sv = Sv.isel(range_sample=search)
    depth = depth.isel(range_sample=search)
    theta = theta.isel(range_sample=search)
    phi = phi.isel(range_sample=search)

    # Smooth split-beam angles
    theta_mask = _smooth_angle(theta, wtheta) ** 2 > ttheta
    phi_mask = _smooth_angle(phi, wphi) ** 2 > tphi
    angle_mask = (theta_mask | phi_mask).compute()

    # Adaptive Sv threshold
    selected_sv = _log2lin(Sv.data[angle_mask.values])

    if isinstance(selected_sv, da.Array):
        selected_sv = selected_sv.compute()

    selected_sv = selected_sv[np.isfinite(selected_sv)]

    if selected_sv.size:
        median_sv = max(
            float(_lin2log(np.median(selected_sv))),
            tSv,
        )

        # Connected components
        sv_mask = (Sv > median_sv).compute()

        components, count = ndima.label(
            sv_mask.values,
            structure=ndima.generate_binary_structure(2, 2),
        )

        # Retain components intersecting the angle mask
        intercepted = np.unique(components[angle_mask.values])
        intercepted = intercepted[intercepted != 0]

        lookup = np.zeros(count + 1, dtype=bool)
        lookup[intercepted] = True
        mask_data = lookup[components]

    else:
        mask_data = np.zeros(Sv.shape, dtype=bool)

    mask = xr.DataArray(
        mask_data,
        dims=Sv.dims,
        coords=Sv.coords,
    )

    # Retrieve ping-dependent bottom depths
    idx = mask.argmax(dim="range_sample")
    bottom_depth = depth.isel(range_sample=idx)
    bottom_depth = bottom_depth.where(mask.any(dim="range_sample")) - offset_m

    return xr.DataArray(
        bottom_depth.data,
        dims=["ping_time"],
        coords={"ping_time": ds["ping_time"]},
        name="bottom_depth",
        attrs={
            "detector": "blackwell",
            "threshold_Sv": tSv,
            "threshold_angle_major": ttheta,
            "threshold_angle_minor": tphi,
            "offset_m": float(offset_m),
            "channel": str(channel),
        },
    )


def _smooth_angle(angle, window):
    """Apply a uniform filter using the underlying array backend."""
    if isinstance(angle.data, da.Array):
        filtered = dask_ndfilters.uniform_filter(
            angle.data,
            size=window,
            mode="reflect",
        )
    else:
        filtered = ndima.uniform_filter(
            angle.data,
            size=window,
            mode="reflect",
        )

    return angle.copy(data=filtered)
