from __future__ import annotations

import pytest

from mn_cofolding.runtime import Settings


@pytest.fixture
def settings(tmp_path):
    root = tmp_path / "portable install"
    return Settings(
        app_home=root / "app",
        runs_dir=root / "results" / "runs",
        reference_dir=root / "reference_files",
        cache_dir=root / "reference_files" / "mn-cofolding" / "cache",
        msa_cache_dir=root / "reference_files" / "msa_cache",
        temp_dir=root / "tmp",
        af3_image="mn-cofolding-af3:test",
        af2_image="mn-colabfold:test",
        docker_executable="docker",
        scheduler_state_dir=root / "scheduler-state",
    )
