import dask.array
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

import echopype as ep
from echopype.mask import detect_seafloor


@pytest.fixture
def ek80_path(test_path):
    return test_path["EK80"]


def _make_ds_Sv_depth(n_ping=100, n_range=40, channel="chan1", dz=1.0, var_name="Sv_corrected"):
    """Build a dataset with Sv and a channelized depth grid."""
    ping = np.arange(n_ping)
    rs = np.arange(n_range)
    depth_1d = rs * dz  # depth increases with range_sample
    depth_2d = np.tile(depth_1d, (n_ping, 1))  # (ping_time, range_sample)

    ds = xr.Dataset()
    ds[var_name] = xr.DataArray(
        data=np.full((1, n_ping, n_range), -120.0, dtype=float),
        dims=("channel", "ping_time", "range_sample"),
        coords={"channel": [channel], "ping_time": ping, "range_sample": rs},
    )
    # make depth have a channel dimension so .sel(channel=...) works
    ds["depth"] = (
        xr.DataArray(
            data=depth_2d,
            dims=("ping_time", "range_sample"),
            coords={"ping_time": ping, "range_sample": rs},
        )
        .expand_dims({"channel": [channel]})
        .transpose("channel", "ping_time", "range_sample")
    )

    return ds


@pytest.mark.unit
def test_detect_basic_return():
    ds = _make_ds_Sv_depth()

    # make a sloped bottom band that gets shallower with time
    n_ping = ds.sizes["ping_time"]
    thickness = 3
    start0 = 22
    slope = -0.02  # bins per ping (negative = shallower with time)

    starts = np.clip(
        start0 + np.round(slope * np.arange(n_ping)).astype(int),
        0,
        ds.sizes["range_sample"] - thickness,
    )

    # fill fake depth with a little noise
    rng = np.random.default_rng(42)
    sigma = 0.7
    for j, s in enumerate(starts):
        noise = rng.normal(0.0, sigma, size=thickness)
        ds["Sv_corrected"].isel(channel=0, ping_time=j, range_sample=slice(s, s + thickness))[:] = (
            -35.0 + noise
        )

    # Call dispatcher (basic)
    basic_depth = detect_seafloor(
        ds,
        method="basic",
        params={
            "var_name": "Sv_corrected",
            "channel": "chan1",
            "threshold": (-50.0, -20.0),
            "offset_m": 0.3,
            "bin_skip_from_surface": 2,
        },
    )

    # Assertions
    assert isinstance(basic_depth, xr.DataArray)
    assert basic_depth.name == "bottom_depth"
    assert basic_depth.dims == ("ping_time",)
    assert np.array_equal(basic_depth["ping_time"], ds["ping_time"])

    idx = xr.DataArray(starts, dims=["ping_time"], coords={"ping_time": ds["ping_time"]})

    # Depth at those first bright bins, one value per ping
    depth_at_starts = ds["depth"].isel(range_sample=idx)

    # Expected detector output (subtract the offset in meters)
    expected_depth = depth_at_starts.values - 0.3

    # Compare to the algorithm’s result
    assert np.allclose(basic_depth.values, expected_depth, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("chunked", [False, True])
def test_detect_basic_no_detection_returns_nan(chunked):
    ds = _make_ds_Sv_depth(
        n_ping=50,
        n_range=30,
        channel="chan1",
        dz=1.0,
        var_name="Sv_corrected",
    )
    ds["Sv_corrected"][:] = -120.0

    if chunked:
        ds = ds.chunk({"ping_time": 10, "range_sample": 5})

    out = detect_seafloor(
        ds,
        method="basic",
        params={
            "var_name": "Sv_corrected",
            "channel": "chan1",
            "threshold": (-50.0, -20.0),
            "offset_m": 0.3,
            "bin_skip_from_surface": 10,
        },
    )

    assert out.dims == ("ping_time",)
    assert np.isnan(out.compute().values).all()


@pytest.mark.unit
def test_detect_basic_dask_input_stays_lazy():
    ds = _make_ds_Sv_depth(
        n_ping=6,
        n_range=20,
        channel="chan1",
        dz=1.0,
        var_name="Sv_corrected",
    )
    ds["Sv_corrected"][:, :, 12] = -35.0
    ds = ds.chunk({"ping_time": 2, "range_sample": 5})

    tasks = []

    with Callback(pretask=lambda key, *args: tasks.append(key)):
        out = detect_seafloor(
            ds,
            method="basic",
            params={
                "var_name": "Sv_corrected",
                "channel": "chan1",
                "threshold": (-50.0, -20.0),
                "offset_m": 0.3,
                "bin_skip_from_surface": 2,
            },
        )

    assert isinstance(out.data, dask.array.Array)
    assert not tasks

    np.testing.assert_allclose(out.compute().values, np.full(6, 11.7), atol=1e-6)


@pytest.mark.unit
def test_detect_blackwell_dask_input():
    ds = _make_ds_Sv_depth(
        n_ping=6,
        n_range=20,
        channel="chan1",
        dz=1.0,
        var_name="Sv_corrected",
    )
    ds["Sv_corrected"][:] = -120.0
    ds["Sv_corrected"][:, :, 12] = -35.0
    ds["angle_alongship"] = xr.full_like(ds["Sv_corrected"], 2.0)
    ds["angle_athwartship"] = xr.zeros_like(ds["Sv_corrected"])

    params = {
        "var_name": "Sv_corrected",
        "channel": "chan1",
        "threshold": (-40.0, 1.0, 1.0),
        "offset_m": 0.3,
        "r0": 0,
        "r1": 19,
        "wtheta": 3,
        "wphi": 3,
    }
    eager = detect_seafloor(ds, method="blackwell", params=params)
    chunked = detect_seafloor(
        ds.chunk({"ping_time": 2, "range_sample": 5}),
        method="blackwell",
        params=params,
    )

    assert isinstance(chunked.data, dask.array.Array)
    np.testing.assert_allclose(chunked.compute().values, np.full(6, 11.7), atol=1e-6)
    np.testing.assert_allclose(chunked.compute().values, eager.values, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("chunked", [False, True])
def test_detect_blackwell_variable_depth(chunked):
    ds = _make_ds_Sv_depth(
        n_ping=6,
        n_range=20,
        channel="chan1",
        dz=1.0,
        var_name="Sv_corrected",
    )

    # Synthetic bottom echo
    ds["Sv_corrected"][:] = -120.0
    ds["Sv_corrected"][:, :, 12:] = -35.0

    ds["angle_alongship"] = xr.full_like(ds["Sv_corrected"], 2.0)
    ds["angle_athwartship"] = xr.zeros_like(ds["Sv_corrected"])

    # Introduce ping-dependent depth variations
    depth_shift = xr.DataArray(
        np.arange(6) * 0.5,
        dims="ping_time",
        coords={"ping_time": ds["ping_time"]},
    )

    ds["depth"] = ds["depth"] + depth_shift

    if chunked:
        ds = ds.chunk({"ping_time": 2, "range_sample": 5})

    result = detect_seafloor(
        ds,
        method="blackwell",
        params={
            "var_name": "Sv_corrected",
            "channel": "chan1",
            "threshold": (-40.0, 1.0, 1.0),
            "offset_m": 0.3,
            "r0": 0,
            "r1": 25,
            "wtheta": 3,
            "wphi": 3,
        },
    )

    expected = 11.7 + np.arange(6) * 0.5

    np.testing.assert_allclose(result.compute().values, expected)

@pytest.mark.unit
def test_detect_seafloor_unknown_method_raises():
    """Verify that an unsupported detection method raises ValueError."""
    ds = _make_ds_Sv_depth()

    with pytest.raises(ValueError, match="Unsupported bottom detection method"):
        detect_seafloor(
            ds,
            method="__bad_method__",
            params={
                "var_name": "Sv_corrected",
                "channel": "59006-125-2",
            },
        )


@pytest.mark.integration
def test_blackwell_vs_basic_close_local(ek80_path):
    """Blackwell vs basic using local test data"""

    raw_path = ek80_path / "ncei-wcsd/SH2306/Hake-D20230811-T165727.raw"

    if not raw_path.is_file():
        pytest.fail(f"Missing EK80 RAW: {raw_path}")

    ed = ep.open_raw(raw_path, sonar_model="EK80")

    ds_Sv = ep.calibrate.compute_Sv(
        ed,
        waveform_mode="CW",
        encode_mode="power",
    )

    ds_Sv = ep.consolidate.add_depth(
        ds_Sv,
        ed,
        depth_offset=9.8,
    )

    # Add split-beam angles required by Blackwell
    ds_Sv = ep.consolidate.add_splitbeam_angle(
        ds_Sv,
        ed,
        waveform_mode="CW",
        encode_mode="power",
        to_disk=False,
    )

    # Pick an available channel
    sel_channel = "WBT 400142-15 ES70-7C_ES"

    blackwell_depth = detect_seafloor(
        ds=ds_Sv,
        method="blackwell",
        params={
            "channel": sel_channel,
            "var_name": "Sv",
            "threshold": [-40, 2.4, 1.0],
            "offset_m": 0.3,
            "r0": 10,
            "r1": 1000,
            "wtheta": 28,
            "wphi": 52,
        },
    )

    basic_depth = detect_seafloor(
        ds=ds_Sv,
        method="basic",
        params={
            "channel": sel_channel,
            "var_name": "Sv",
            "threshold": (-40, -20),
            "offset_m": 0.3,
            "bin_skip_from_surface": 200,
        },
    )

    assert isinstance(blackwell_depth, xr.DataArray)
    assert blackwell_depth.dims == ("ping_time",)

    assert isinstance(basic_depth, xr.DataArray)
    assert basic_depth.dims == ("ping_time",)

    assert np.array_equal(
        blackwell_depth["ping_time"],
        basic_depth["ping_time"],
    )

    b = blackwell_depth.values
    c = basic_depth.values

    m = np.isfinite(b) & np.isfinite(c)
    assert m.any(), "No overlapping finite detections to compare."

    mad = np.median(np.abs(b[m] - c[m]))
    assert mad < 5.0, f"Median abs diff too large: {mad:.2f} m"


@pytest.mark.unit
@pytest.mark.parametrize("chunked", [False, True])
def test_detect_basic_variable_depth(chunked):
    ds = xr.Dataset(
        {
            "Sv": (
                ("channel", "ping_time", "range_sample"),
                [[[-80.0, -30.0, -25.0], [-80.0, -70.0, -30.0]]],
            ),
            "depth": (
                ("channel", "ping_time", "range_sample"),
                [[[10.0, 20.0, 30.0], [12.0, 22.0, 32.0]]],
            ),
        },
        coords={
            "channel": ["test"],
            "ping_time": [0, 1],
            "range_sample": [0, 1, 2],
        },
    )

    if chunked:
        ds = ds.chunk({"ping_time": 1, "range_sample": 2})

    result = detect_seafloor(
        ds,
        method="basic",
        params={
            "var_name": "Sv",
            "channel": "test",
            "threshold": (-40, -20),
            "offset_m": 0.5,
            "bin_skip_from_surface": 0,
        },
    )

    np.testing.assert_allclose(result.compute().values, [19.5, 31.5])
