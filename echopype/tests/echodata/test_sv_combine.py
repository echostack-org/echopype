
"""Test combining calibrated Sv datasets from multiple files using Xarray and Zarr"""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

import echopype as ep


@pytest.fixture
def sv_parts():
    """Create small synthetic Sv datasets with increasing ping times"""
    parts = []

    for i in range(3):
        times = np.arange(i * 2, i * 2 + 2).astype("timedelta64[s]") + np.datetime64(
            "2026-01-01"
        )

        parts.append(
            xr.Dataset(
                {
                    "Sv": (
                        ("channel", "ping_time", "range_sample"),
                        np.full((2, 2, 3), -65.0 + i),
                    )
                },
                coords={
                    "channel": ["ch1", "ch2"],
                    "ping_time": times,
                    "range_sample": [0, 1, 2],
                },
            )
        )

    return parts


def _write_open_combine(parts, tmp_path):
    """Save individual Sv datasets to Zarr and reopen them as one combined dataset"""
    paths = []

    for index, part in enumerate(parts):
        path = tmp_path / f"Sv_{index}.zarr"
        part.to_zarr(path, mode="w")
        paths.append(str(path))

    return xr.open_mfdataset(
        paths,
        engine="zarr",
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        join="exact",
        compat="override",
    )


def _save_combined(dataset, path):
    """Clear inherited chunk encoding and rechunk the combined dataset before saving to Zarr"""
    dataset = dataset.drop_encoding()

    dataset = dataset.chunk(
        {
            "channel": 1,
            "ping_time": min(24, dataset.sizes["ping_time"]),
            "range_sample": min(500, dataset.sizes["range_sample"]),
        }
    )

    dataset.to_zarr(path, mode="w")


@pytest.mark.parametrize("n_parts", [2, 3])
def test_sv_zarr_multi_file_combination(tmp_path, sv_parts, n_parts):
    """Check that multiple Sv datasets can be combined, saved, and reopened without changing their values"""
    parts = sv_parts[:n_parts]
    expected = xr.concat(parts, dim="ping_time")

    with _write_open_combine(parts, tmp_path) as combined:
        assert_allclose(combined["Sv"], expected["Sv"])
        assert combined.sizes["ping_time"] == n_parts * 2

        output = tmp_path / "combined_Sv.zarr"
        _save_combined(combined, output)

    with xr.open_zarr(output) as reopened:
        assert_allclose(reopened["Sv"], expected["Sv"])


def test_sv_combine_rejects_incompatible_channels(tmp_path, sv_parts):
    """Check that different channels cannot be combined directly"""
    left, right = sv_parts[:2]
    right = right.assign_coords(channel=["other1", "other2"])

    with pytest.raises(ValueError):
        with _write_open_combine([left, right], tmp_path):
            pass


def test_sv_channel_subset(tmp_path, sv_parts):
    """Check combining datasets with shared channels"""
    left, right = sv_parts[:2]
    right = right.assign_coords(channel=["ch1", "other"])

    common = [
        channel
        for channel in left.channel.values.tolist()
        if channel in right.channel.values
    ]

    assert common == ["ch1"]

    selected = [part.sel(channel=common) for part in (left, right)]

    with _write_open_combine(selected, tmp_path) as combined:
        assert combined.channel.values.tolist() == common
        assert_allclose(
            combined["Sv"],
            xr.concat(selected, dim="ping_time")["Sv"],
        )


def test_sv_different_range_sample_lengths(tmp_path, sv_parts):
    """Check combining datasets with different sample lengths"""
    left, right = sv_parts[:2]
    right = right.isel(range_sample=slice(None, 2))

    common_samples = left.range_sample.values
    normalized = [
        part.reindex(range_sample=common_samples)
        for part in (left, right)
    ]

    with _write_open_combine(normalized, tmp_path) as combined:
        assert combined.sizes["range_sample"] == 3
        assert combined.sizes["ping_time"] == 4

        assert bool(
            combined["Sv"]
            .isel(ping_time=slice(2, None), range_sample=2)
            .isnull()
            .all()
        )

        assert_allclose(
            combined["Sv"],
            xr.concat(normalized, dim="ping_time")["Sv"],
        )


@pytest.fixture(
    params=[
        pytest.param(
            (
                "EK60",
                "EK60",
                [
                    ("ncei-wcsd", "Summer2017-D20170620-T011027.raw"),
                    ("ncei-wcsd", "Summer2017-D20170620-T014302.raw"),
                ],
                None,
                None,
                None,
            ),
            id="ek60",
        ),
        pytest.param(
            (
                "EK60",
                "EK60",
                [
                    ("ncei-wcsd", "SH1701", "TEST-D20170114-T202932.raw"),
                    ("ncei-wcsd", "SH1701", "TEST-D20170114-T203337.raw"),
                ],
                None,
                None,
                None,
            ),
            id="ek60-different-samples",
        ),
        pytest.param(
            (
                "AZFP",
                "AZFP",
                [
                    ("ooi", "18100407.01A"),
                    ("ooi", "18100408.01A"),
                ],
                ("ooi", "18092920.XML"),
                None,
                None,
            ),
            id="azfp",
        ),
        pytest.param(
            (
                "EK80",
                "EK80",
                [
                    ("ncei-wcsd", "SH1707", "Reduced_D20170826-T205615.raw"),
                    ("ncei-wcsd", "SH1707", "Reduced_D20170826-T205659.raw"),
                ],
                None,
                "BB",
                "complex",
            ),
            id="ek80-broadband",
        ),
        pytest.param(
            (
                "EK80",
                "EK80",
                [
                    (
                        "ncei-wcsd",
                        "SH2106",
                        "EK80",
                        "Reduced_Hake-D20210701-T130426.raw",
                    ),
                    (
                        "ncei-wcsd",
                        "SH2106",
                        "EK80",
                        "Reduced_Hake-D20210701-T131325.raw",
                    ),
                ],
                None,
                "auto-CW",
                None,
            ),
            id="ek80-different-samples",
        ),
    ]
)
def raw_sv_case(request, test_path):
    """Reuse EK60, EK80, and AZFP raw files from the original EchoData combination tests"""
    key, sonar_model, names, xml, waveform_mode, encode_mode = request.param

    root = test_path[key]
    raw_paths = [root.joinpath(*segments) for segments in names]
    xml_path = root.joinpath(*xml) if xml is not None else None

    return raw_paths, sonar_model, xml_path, waveform_mode, encode_mode


def _cw_encoding(echodata):
    """Find the CW encoding used by the file"""
    descriptions = echodata["Sonar"]["waveform_encode_descr"].values.tolist()

    if "complex_CW" in descriptions:
        return "complex"

    if "power" in descriptions:
        return "power"

    pytest.fail(f"expected CW data but found {descriptions}")


@pytest.mark.integration
def test_real_instrument_sv_zarr_combination(raw_sv_case, tmp_path):
    """Check combining calibrated Sv from real raw files"""
    paths, sonar_model, xml_path, waveform_mode, encode_mode = raw_sv_case

    sv_datasets = []

    for path in paths:
        open_kwargs = {"sonar_model": sonar_model}

        if xml_path is not None:
            open_kwargs["xml_path"] = xml_path

        ed = ep.open_raw(path, **open_kwargs)

        cal_kwargs = {}

        if waveform_mode == "auto-CW":
            cal_kwargs["waveform_mode"] = "CW"
            cal_kwargs["encode_mode"] = _cw_encoding(ed)

        elif waveform_mode is not None:
            cal_kwargs["waveform_mode"] = waveform_mode
            cal_kwargs["encode_mode"] = encode_mode

        # these azfp files need salinity and pressure
        if sonar_model == "AZFP":
            cal_kwargs["env_params"] = {
                "salinity": 35.0,
                "pressure": 10.0,
            }

        calibrated = ep.calibrate.compute_Sv(ed, **cal_kwargs)
        sv_datasets.append(calibrated[["Sv"]])

    # keep only channels shared by all files
    common_channels = sv_datasets[0].channel.values.tolist()

    for ds in sv_datasets[1:]:
        common_channels = [
            ch for ch in common_channels if ch in ds.channel.values
        ]

    assert common_channels, "no shared channels across raw files"

    # pad shorter sample ranges before combining
    max_sample = max(ds.sizes["range_sample"] for ds in sv_datasets)
    sample_index = np.arange(max_sample)

    normalized = [
        ds.sel(channel=common_channels).reindex(range_sample=sample_index)
        for ds in sv_datasets
    ]

    # check that ping times are in order
    for earlier, later in zip(normalized, normalized[1:]):
        assert earlier.ping_time.max().values < later.ping_time.min().values

    # save and combine the calibrated datasets
    with _write_open_combine(normalized, tmp_path) as combined:
        expected = xr.concat(normalized, dim="ping_time", join="exact")

        assert combined.sizes["ping_time"] == sum(
            ds.sizes["ping_time"] for ds in normalized
        )

        assert_allclose(combined["Sv"], expected["Sv"])

        out = tmp_path / "combined_Sv.zarr"
        _save_combined(combined, out)

    # check the saved result
    with xr.open_zarr(out) as reopened:
        assert_allclose(reopened["Sv"], expected["Sv"])

        assert reopened.sizes["ping_time"] == sum(
            ds.sizes["ping_time"] for ds in normalized
        )
