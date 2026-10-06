from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import AF2_MODELS, AF3_MODELS, get_model
from .runtime import Settings, get_settings


@dataclass(frozen=True)
class ReferenceStatus:
    model: str
    family: str
    state: str
    path: str
    detail: str


def reference_status(model_name: str, settings: Settings | None = None) -> ReferenceStatus:
    settings = settings or get_settings()
    model = get_model(model_name)
    if model.family == "af2":
        root = settings.reference_dir / "alphafold_models"
        suffix = "ptm" if model_name == "af2_ptm" else "multimer_v3"
        files = [root / f"params_model_{i}_{suffix}.npz" for i in range(1, 6)]
        present = sum(path.is_file() for path in files)
        state = "ready" if present == 5 else "partial" if present else "missing"
        return ReferenceStatus(model_name, model.family, state, str(root), f"{present}/5 parameter files found")
    if model_name == "alphafold3":
        root = settings.reference_dir / "alphafold3"
        present = (root / "af3.bin.zst").is_file()
        return ReferenceStatus(
            model_name,
            model.family,
            "ready" if present else "missing",
            str(root),
            "Existing official af3.bin.zst can be reused" if present else "Official parameters need to be supplied under the published terms",
        )
    if model_name == "chai1":
        root = settings.reference_dir / "chai1"
        required = [
            root / "models_v2" / f"{name}.pt"
            for name in (
                "feature_embedding",
                "bond_loss_input_proj",
                "token_embedder",
                "trunk",
                "diffusion_module",
                "confidence_head",
            )
        ]
        required.extend(
            [
                root / "conformers_v1.apkl",
                root / "esm" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt",
            ]
        )
        present = sum(path.is_file() for path in required)
        state = "ready" if present == len(required) else "partial" if present else "missing"
        return ReferenceStatus(
            model_name,
            model.family,
            state,
            str(root),
            f"{present}/{len(required)} pinned native Chai-1 files found (separate from AF3 Chai int8 weights)",
        )
    weights_root = settings.cache_dir / "alphafold3" / "weights"
    candidates = []
    for precision in ("int8", "fp16", "fp32"):
        suffix = "" if precision == "fp32" else f"-{precision}"
        filename = f"{model_name}.bin.zst" if precision == "fp32" else f"{model_name}.{precision}.bin.zst"
        candidates.append(weights_root / f"{model_name}{suffix}" / filename)
    present = next((path for path in candidates if path.is_file()), None)
    root = present.parent if present else weights_root
    return ReferenceStatus(
        model_name,
        model.family,
        "ready" if present else "on-demand",
        str(root),
        f"Cached preview weights found at {present}" if present else "Missing preview-format weights are fetched on first use; MSA results, ESM towers, and JAX caches use the configured shared cache",
    )


def all_reference_statuses(settings: Settings | None = None) -> list[ReferenceStatus]:
    settings = settings or get_settings()
    return [reference_status(model.name, settings) for model in (*AF3_MODELS, *AF2_MODELS)]
