import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonicsphere.config import get_settings  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def cipic(settings):
    from sonicsphere.dataset_loader import DatasetError, get_dataset
    try:
        return get_dataset("cipic", settings.dataset("cipic").default_subject, settings)
    except DatasetError as exc:
        pytest.skip(f"CIPIC not downloaded: {exc}")


@pytest.fixture(scope="session")
def prepared(settings, cipic):
    from sonicsphere.filter_pack import get_prepared
    return get_prepared("cipic", cipic.subject, None, settings)


@pytest.fixture(scope="session")
def interp(prepared, settings):
    from sonicsphere.hrir_interpolation import HRIRInterpolator
    r = settings.rendering
    return HRIRInterpolator(prepared.filters, "barycentric_aligned", r.gap_edge_threshold_deg, r.measured_snap_deg,
                            triangulation=prepared.triangulation, direction_quantum_deg=r.direction_quantum_deg)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
