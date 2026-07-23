from pathlib import Path

import numpy as np
import pytest

from echopype import open_raw

pytestmark = pytest.mark.integration

BI500_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent / "test_data" / "bi500"
)


@pytest.fixture
def bi500_path():
    if not BI500_DATA_DIR.is_dir():
        pytest.skip(f"BI500 test data not found at {BI500_DATA_DIR}")
    return BI500_DATA_DIR


def test_convert_bi500(bi500_path):
    """Verify BI500 raw files convert to EchoData with expected shapes."""
    echodata = open_raw(raw_file=str(bi500_path), sonar_model="BI500")

    assert echodata.sonar_model == "BI500"
    assert len(echodata["Sonar/Beam_group1"].channel) == 1

    backscatter_r = echodata["Sonar/Beam_group1"].backscatter_r
    backscatter_r_bottom = echodata["Sonar/Beam_group1"].backscatter_r_bottom

    assert backscatter_r.shape == (1, 3323, 500)
    assert backscatter_r_bottom.shape == (1, 3323, 150)
    assert echodata["Platform"].ping_time.shape == (3323,)
    assert echodata["Vendor_specific"].target_depth.shape[0] == 8724

    assert echodata["Sonar/Beam_group1"].channel.values[0] == "BI500-F11990-T01"
    assert echodata["Provenance"].nation_code.values == 31
    assert echodata["Provenance"].ship_code.values == 445
    assert echodata["Provenance"].survey_code.values == 2000008

    assert np.isfinite(backscatter_r.values).any()
    assert np.isfinite(backscatter_r_bottom.values).any()
