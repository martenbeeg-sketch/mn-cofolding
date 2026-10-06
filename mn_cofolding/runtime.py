from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_APP_HOME = Path("/mnt/data/RESULTS/mn-cofolding-workdir")
DEFAULT_REFERENCE_ROOT = Path("/mnt/db/reference_files")


@dataclass(frozen=True)
class Settings:
    app_home: Path
    runs_dir: Path
    reference_dir: Path
    cache_dir: Path
    temp_dir: Path
    af3_image: str
    af2_image: str
    docker_executable: str
    esmfold2_image: str = "mn-esmfold2-opt:0.1.0-cu130"
    chai1_image: str = "mn-chai1-opt:0.6.1-cu130"
    scheduler_state_dir: Path | None = None
    msa_cache_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        home = Path(os.getenv("MN_COFOLDING_APP_HOME", str(DEFAULT_APP_HOME))).expanduser().resolve()
        runs = Path(os.getenv("MN_COFOLDING_RUN_DIR", str(home / "workdir" / "runs"))).expanduser().resolve()
        ref_value = os.getenv("MN_COFOLDING_REFERENCE_DIR")
        if ref_value:
            reference = Path(ref_value).expanduser().resolve()
        elif DEFAULT_REFERENCE_ROOT.is_dir():
            reference = DEFAULT_REFERENCE_ROOT.resolve()
        else:
            reference = (home / "reference_files").resolve()
        cache = Path(os.getenv("MN_COFOLDING_CACHE_DIR", str(reference / "mn-cofolding" / "cache"))).expanduser().resolve()
        msa_cache_value = (
            os.getenv("MN_SHARED_MSA_CACHE_DIR")
            or os.getenv("MN_COFOLDING_MSA_CACHE_DIR")
            or str(reference / "msa_cache")
        )
        msa_cache = Path(msa_cache_value).expanduser().resolve()
        temp = Path(os.getenv("MN_COFOLDING_TMPDIR", str(home / "tmp"))).expanduser().resolve()
        state_value = os.getenv("MN_COMPUTE_SCHEDULER_STATE_DIR")
        return cls(
            app_home=home,
            runs_dir=runs,
            reference_dir=reference,
            cache_dir=cache,
            temp_dir=temp,
            # All AF3 modes, including ``off``, run through this single image.
            # The stock image/tag remains available as a manual fallback.
            af3_image=os.getenv("MN_COFOLDING_AF3_IMAGE", "mn-cofolding-af3-memory:3.1.14-cu13"),
            af2_image=os.getenv("MN_COFOLDING_AF2_IMAGE", "mn-colabfold:1.6.1-cu12"),
            esmfold2_image=os.getenv("MN_COFOLDING_ESMFOLD2_IMAGE", "mn-esmfold2-opt:0.1.0-cu130"),
            chai1_image=os.getenv("MN_COFOLDING_CHAI1_IMAGE", "mn-chai1-opt:0.6.1-cu130"),
            docker_executable=os.getenv("MN_COFOLDING_DOCKER", "docker"),
            scheduler_state_dir=Path(state_value).expanduser().resolve() if state_value else None,
            msa_cache_dir=msa_cache,
        )

    def ensure_dirs(self) -> None:
        self.app_home.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.msa_cache_dir or (self.reference_dir / "msa_cache")).mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        (self.app_home / "workdir").mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings.from_env()
