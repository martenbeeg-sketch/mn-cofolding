from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_AF3_INFERENCE_RE = re.compile(
    r"Running model inference with seed (?P<seed>\d+) took (?P<seconds>[\d.]+) seconds\."
)
_AF3_EXTRACTION_RE = re.compile(
    r"Extracting (?P<samples>\d+) inference samples with seed (?P<seed>\d+) took (?P<seconds>[\d.]+) seconds\."
)
_AF2_ATTEMPT_RE = re.compile(
    r"(?P<model>[A-Za-z0-9_.-]+) took (?P<seconds>[\d.]+)s(?: \((?P<recycles>\d+) recycles\))?"
)
_BOLTZ_PHASE_RE = re.compile(
    r"PHASE item=(?P<item>\S+).*?total_s=(?P<seconds>[\d.]+)"
)
_AF3_SAMPLE_DIR_RE = re.compile(r"seed-(?P<seed>\d+)_sample-(?P<sample>\d+)")
_AF2_RANK_RE = re.compile(r"(?:unrelaxed_|relaxed_|scores_)rank_(?P<rank>\d+)_?(?P<tail>.*)")
_ESMFOLD2_SAMPLE_RE = re.compile(r"__s(?P<seed>\d+)_x(?P<sample>\d+)$")
_CHAI1_SEED_RE = re.compile(r"seed_(?P<seed>\d+)")
_CHAI1_SAMPLE_RE = re.compile(r"model_idx_(?P<sample>\d+)$")
_CHAI1_FORWARD_RE = re.compile(
    r"\[chai1-opt(?: stock)?\] FORWARD item=\S+ out=seed_(?P<seed>\d+).*?forward_s=(?P<seconds>[\d.]+)"
)


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "—"
    value = float(seconds)
    if value < 60:
        return f"{value:.1f} s"
    whole = int(value)
    minutes, remainder = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes} min {remainder} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min {remainder} s"


def _timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def job_durations(metadata: dict, result: dict, *, now: datetime | None = None) -> dict[str, float | None]:
    """Return wall, queue/setup, and engine durations when their timestamps exist."""
    created = _timestamp(metadata.get("created_at"))
    started = _timestamp(metadata.get("started_at"))
    finished = _timestamp(result.get("finished_at") or metadata.get("completed_at"))
    if finished is None and metadata.get("status") not in {"completed", "failed", "cancelled"}:
        finished = now or datetime.now(timezone.utc)
    if created is None:
        return {"total": None, "queue_and_setup": None, "engine": None}
    total = max(0.0, (finished - created).total_seconds()) if finished else None
    queue_and_setup = max(0.0, (started - created).total_seconds()) if started else None
    engine = max(0.0, (finished - started).total_seconds()) if started and finished else None
    return {"total": total, "queue_and_setup": queue_and_setup, "engine": engine}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _flatten_numbers(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []


def _float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _numeric_matrix(value: object) -> list[list[float]] | None:
    """Accept only finite, square numeric matrices from model output JSON."""
    if not isinstance(value, list) or not value:
        return None
    size = len(value)
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            return None
        converted = []
        for item in row:
            number = _float(item)
            if number is None:
                return None
            converted.append(number)
        matrix.append(converted)
    return matrix


def _prediction_structures(run_dir: Path) -> list[Path]:
    output_dir = run_dir / "output"
    structures = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".cif", ".mmcif", ".pdb"}
    )
    af3_samples = [
        path
        for path in structures
        if any(_AF3_SAMPLE_DIR_RE.fullmatch(parent.name) for parent in path.parents)
    ]
    if af3_samples:
        structures = af3_samples

    chai1_samples = [
        path
        for path in structures
        if any(_CHAI1_SEED_RE.fullmatch(parent.name) for parent in path.parents)
    ]
    if chai1_samples:
        structures = chai1_samples

    # ColabFold can emit both relaxed and unrelaxed copies of one ranked model.
    # Prefer the relaxed model when present, while keeping separate seeds/ranks.
    by_rank: dict[str, Path] = {}
    unclassified: list[Path] = []
    for path in structures:
        match = _AF2_RANK_RE.search(path.stem)
        if not match:
            unclassified.append(path)
            continue
        key = f"rank-{match.group('rank')}-{match.group('tail')}"
        previous = by_rank.get(key)
        if previous is None or ("relaxed" in path.stem and "relaxed" not in previous.stem):
            by_rank[key] = path
    return sorted([*unclassified, *by_rank.values()])


def _associated_json(structure: Path, suffix: str) -> dict[str, Any]:
    stem = structure.stem
    if stem.endswith("_model"):
        prefix = stem.removesuffix("_model")
        candidates = [structure.with_name(f"{prefix}_{suffix}.json")]
    else:
        candidates = []
        for marker in ("_unrelaxed_rank_", "_relaxed_rank_"):
            if marker in stem:
                prefix = stem.replace(marker, "_scores_rank_", 1)
                candidates.append(structure.with_name(f"{prefix}.json"))
        candidates.extend(
            path
            for path in structure.parent.glob("*_scores_*.json")
            if any(token in path.stem for token in ("_rank_", "_seed_"))
        )
    for candidate in candidates:
        if candidate.is_file():
            payload = _read_json(candidate)
            if payload:
                return payload
    return {}


def _esmfold2_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "output" / "pred_rows.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("pred_file"):
            rows[str(row["pred_file"])] = row
    return rows


def _npz_array(path: Path, name: str):
    """Load one numeric array from an engine confidence NPZ, if present."""
    if not path.is_file():
        return None
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            if name not in archive.files:
                return None
            values = np.asarray(archive[name], dtype=float)
        if values.size and not np.isfinite(values).all():
            return None
        return values
    except (ImportError, OSError, ValueError, EOFError):
        return None


def _chai1_scores(structure: Path) -> dict[str, Any]:
    match = _CHAI1_SAMPLE_RE.search(structure.stem)
    if not match:
        return {}
    path = structure.with_name(f"scores.model_idx_{match.group('sample')}.npz")
    if not path.is_file():
        return {}
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            return {
                key: np.asarray(archive[key]).tolist()
                for key in archive.files
                if np.issubdtype(np.asarray(archive[key]).dtype, np.number)
                or np.asarray(archive[key]).dtype == np.bool_
            }
    except (ImportError, OSError, ValueError, EOFError):
        return {}


def _boltz_output(
    structure: Path,
) -> tuple[dict[str, Any], list[float], list[list[float]] | None, list[list[float]] | None, int | None, float | None]:
    """Read Boltz-2's confidence JSON and compact its large confidence maps for display."""
    stem = structure.stem
    confidence = _read_json(structure.with_name(f"confidence_{stem}.json"))
    if not confidence:
        return {}, [], None, None, None, None

    plddt_array = _npz_array(structure.with_name(f"plddt_{stem}.npz"), "plddt")
    plddt_values: list[float] = []
    if plddt_array is not None and plddt_array.ndim == 1:
        scale = 100.0 if plddt_array.size and float(plddt_array.max()) <= 1.0 else 1.0
        plddt_values = (plddt_array * scale).tolist()

    pae_array = _npz_array(structure.with_name(f"pae_{stem}.npz"), "pae")
    pae: list[list[float]] | None = None
    pae_preview: list[list[float]] | None = None
    pae_residue_count: int | None = None
    pae_max: float | None = None
    if pae_array is not None and pae_array.ndim == 2 and pae_array.shape[0] == pae_array.shape[1]:
        import numpy as np

        pae_residue_count = int(pae_array.shape[0])
        pae_max = float(np.max(pae_array)) if pae_array.size else None
        if pae_residue_count <= 1024:
            pae = pae_array.tolist()
            pae_preview = pae
        else:
            # Keep the full NPZ in the job files, but avoid expanding a 4k x 4k
            # matrix into millions of Python floats or an enormous viewer page.
            indices = np.linspace(0, pae_residue_count - 1, 512).round().astype(int)
            pae_preview = pae_array[np.ix_(indices, indices)].tolist()

    return confidence, plddt_values, pae, pae_preview, pae_residue_count, pae_max


def _prediction_label(structure: Path, run_dir: Path) -> tuple[str, str | None, int | None]:
    sample_match = next(
        (_AF3_SAMPLE_DIR_RE.fullmatch(parent.name) for parent in structure.parents if _AF3_SAMPLE_DIR_RE.fullmatch(parent.name)),
        None,
    )
    relative = structure.relative_to(run_dir).as_posix()
    if sample_match:
        seed = int(sample_match.group("seed"))
        sample = int(sample_match.group("sample"))
        return f"Seed {seed}, sample {sample + 1}", f"seed-{seed}", sample
    rank_match = _AF2_RANK_RE.search(structure.stem)
    if rank_match:
        model = rank_match.group("tail").replace("_", " ").strip()
        return f"Rank {int(rank_match.group('rank')):03d} · {model or structure.stem}", None, None
    esm_match = _ESMFOLD2_SAMPLE_RE.search(structure.stem)
    if esm_match:
        seed = int(esm_match.group("seed"))
        sample = int(esm_match.group("sample"))
        return f"Seed {seed}, sample {sample + 1}", f"seed-{seed}", sample
    chai_seed = next(
        (_CHAI1_SEED_RE.fullmatch(parent.name) for parent in structure.parents if _CHAI1_SEED_RE.fullmatch(parent.name)),
        None,
    )
    chai_sample = _CHAI1_SAMPLE_RE.search(structure.stem)
    if chai_seed and chai_sample:
        seed = int(chai_seed.group("seed"))
        sample = int(chai_sample.group("sample"))
        return f"Seed {seed}, sample {sample + 1}", f"seed-{seed}", sample
    return structure.stem, None, None


def prediction_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """Collect structure files and the confidence data written beside each one."""
    root = Path(run_dir).resolve()
    esmfold2_rows = _esmfold2_rows(root)
    records: list[dict[str, Any]] = []
    for structure in _prediction_structures(root):
        boltz_confidence_path = structure.with_name(f"confidence_{structure.stem}.json")
        is_boltz = boltz_confidence_path.is_file()
        is_esmfold2 = structure.name in esmfold2_rows or bool(_ESMFOLD2_SAMPLE_RE.search(structure.stem))
        chai_scores = _chai1_scores(structure)
        is_chai1 = bool(chai_scores)
        pae_preview = None
        pae_residue_count = None
        boltz_pae_max = None
        if is_boltz:
            summary, boltz_plddt, pae, pae_preview, pae_residue_count, boltz_pae_max = _boltz_output(structure)
            confidence = {}
        elif is_esmfold2:
            summary = esmfold2_rows.get(structure.name, {})
            confidence = {}
            plddt_array = _npz_array(structure.with_name(f"{structure.stem}_pae.npz"), "plddt")
            boltz_plddt = []
            if plddt_array is not None:
                import numpy as np

                values = np.asarray(plddt_array, dtype=float).reshape(-1)
                scale = 100.0 if values.size and float(values.max()) <= 1.0 else 1.0
                boltz_plddt = (values * scale).tolist()
            pae_array = _npz_array(structure.with_name(f"{structure.stem}_pae.npz"), "pae")
            pae = None
            pae_preview = None
            pae_residue_count = None
            boltz_pae_max = None
            if pae_array is not None and pae_array.ndim == 2 and pae_array.shape[0] == pae_array.shape[1]:
                import numpy as np

                pae_residue_count = int(pae_array.shape[0])
                boltz_pae_max = float(np.max(pae_array)) if pae_array.size else None
                if pae_residue_count <= 1024:
                    pae = pae_array.tolist()
                    pae_preview = pae
                else:
                    indices = np.linspace(0, pae_residue_count - 1, 512).round().astype(int)
                    pae_preview = pae_array[np.ix_(indices, indices)].tolist()
        elif is_chai1:
            summary = chai_scores
            confidence = chai_scores
            boltz_plddt = _flatten_numbers(chai_scores.get("plddt") or chai_scores.get("atom_plddts"))
            if boltz_plddt and max(boltz_plddt) <= 1.0:
                boltz_plddt = [value * 100.0 for value in boltz_plddt]
            pae_value = chai_scores.get("pae") or chai_scores.get("predicted_aligned_error")
            pae = _numeric_matrix(pae_value)
            pae_preview = pae
            pae_residue_count = len(pae) if pae else None
            boltz_pae_max = max(_flatten_numbers(pae_value), default=None)
        elif structure.stem.endswith("_model"):
            summary = _associated_json(structure, "summary_confidences")
            confidence = _associated_json(structure, "confidences")
            boltz_plddt = []
            pae = None
        else:
            summary = {}
            confidence = _associated_json(structure, "scores")
            boltz_plddt = []
            pae = None
        combined = {**confidence, **summary}
        plddt_values = boltz_plddt or _flatten_numbers(combined.get("atom_plddts") or combined.get("plddt"))
        # The confidence-disabled ESMFold2+ESM-C checkpoints write all-zero
        # pLDDT arrays as placeholders. Treat those as absent, not as a real
        # confidence score of zero.
        if plddt_values and not any(value > 0 for value in plddt_values):
            plddt_values = []
        if not is_boltz and not is_esmfold2 and not is_chai1:
            pae = _numeric_matrix(combined.get("pae") or combined.get("predicted_aligned_error"))
            pae_preview = pae
            pae_residue_count = len(pae) if pae else None
        contact_probs = _numeric_matrix(combined.get("contact_probs"))
        pae_values = _flatten_numbers(pae)
        if pae_preview:
            pae_values.extend(_flatten_numbers(pae_preview))
        label, seed, sample = _prediction_label(structure, root)
        records.append(
            {
                "label": label,
                "relative_path": structure.relative_to(root).as_posix(),
                "structure_path": structure,
                "seed": seed,
                "sample": sample,
                "mean_plddt": (
                    sum(plddt_values) / len(plddt_values)
                    if plddt_values
                    else _float(combined.get("complex_plddt")) * 100
                    if is_boltz and _float(combined.get("complex_plddt")) is not None
                    else None
                ),
                "ptm": _float(combined.get("ptm")),
                "iptm": _float(combined.get("iptm")),
                "ranking_score": _float(combined.get("ranking_score") or combined.get("confidence_score") or combined.get("aggregate_score")),
                "pae": pae,
                "pae_plot_values": pae_preview,
                "pae_downsampled": bool((is_boltz or is_esmfold2 or is_chai1) and pae_residue_count and pae_residue_count > 1024),
                "pae_residue_count": pae_residue_count,
                "contact_probs": contact_probs,
                "max_pae": _float(
                    combined.get("max_pae")
                    or combined.get("max_predicted_aligned_error")
                    or boltz_pae_max
                    or (max(pae_values) if pae_values else None)
                ),
                "plddt_values": plddt_values,
                "fraction_disordered": _float(combined.get("fraction_disordered")),
                "has_clash": _float(combined.get("has_clash")),
            }
        )
    return records


def parse_attempt_timings(
    run_dir: str | Path,
    family: str,
    *,
    samples_per_seed: int = 1,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Read per-seed AF3 or per-model/seed AF2 timings from the engine logs."""
    root = Path(run_dir)
    if model_name == "boltz2":
        result = _read_json(root / "result.json")
        if result and not result.get("success", False):
            # Boltz can log a short PHASE record after catching an OOM. It is
            # not a completed model inference time.
            return []
        texts = []
        for name in ("stdout.log", "stderr.log"):
            path = root / name
            if path.is_file():
                texts.append(path.read_text(errors="replace"))
        attempts = []
        for match in _BOLTZ_PHASE_RE.finditer("\n".join(texts)):
            attempts.append(
                {
                    "Attempt": f"{match.group('item')} · Boltz2 inference",
                    "Inference time": float(match.group("seconds")),
                    "Output processing time": None,
                    "Predictions": 1,
                    "Recycles": None,
                }
            )
        return attempts

    if model_name == "chai1":
        result = _read_json(root / "result.json")
        if result and not result.get("success", False):
            return []
        texts = []
        for name in ("stdout.log", "stderr.log"):
            path = root / name
            if path.is_file():
                texts.append(path.read_text(errors="replace"))
        return [
            {
                "Attempt": f"Seed {int(match.group('seed'))}",
                "Inference time": float(match.group("seconds")),
                "Output processing time": None,
                "Predictions": samples_per_seed,
                "Recycles": None,
            }
            for match in _CHAI1_FORWARD_RE.finditer("\n".join(texts))
        ]

    if model_name == "esmfold2":
        result = _read_json(root / "result.json")
        if result and not result.get("success", False):
            return []
        seed_times: dict[int, float] = {}
        for row in _esmfold2_rows(root).values():
            seed = _float(row.get("seed"))
            seconds = _float(row.get("wall_s"))
            if seed is None:
                match = _ESMFOLD2_SAMPLE_RE.search(str(row.get("pred_file") or ""))
                if match:
                    seed = float(match.group("seed"))
            if seed is not None and seconds is not None:
                seed_times.setdefault(int(seed), seconds)
        return [
            {
                "Attempt": f"Seed {seed}",
                "Inference time": seconds,
                "Output processing time": None,
                "Predictions": samples_per_seed,
                "Recycles": None,
            }
            for seed, seconds in sorted(seed_times.items())
        ]

    if family == "af2":
        candidates = [root / "output" / "log.txt", root / "stdout.log", root / "stderr.log"]
        text = next((path.read_text(errors="replace") for path in candidates if path.is_file() and path.stat().st_size), "")
        attempts = []
        for match in _AF2_ATTEMPT_RE.finditer(text):
            attempts.append(
                {
                    "Attempt": match.group("model"),
                    "Inference time": float(match.group("seconds")),
                    "Predictions": 1,
                    "Recycles": int(match.group("recycles")) if match.group("recycles") else None,
                }
            )
        return attempts

    texts = []
    for name in ("stdout.log", "stderr.log"):
        path = root / name
        if path.is_file():
            texts.append(path.read_text(errors="replace"))
    text = "\n".join(texts)
    inference_by_seed: dict[int, float] = {}
    extraction_by_seed: dict[int, tuple[float, int]] = {}
    for match in _AF3_INFERENCE_RE.finditer(text):
        inference_by_seed[int(match.group("seed"))] = float(match.group("seconds"))
    for match in _AF3_EXTRACTION_RE.finditer(text):
        extraction_by_seed[int(match.group("seed"))] = (
            float(match.group("seconds")),
            int(match.group("samples")),
        )
    attempts = []
    for seed in sorted(set(inference_by_seed) | set(extraction_by_seed)):
        extraction_seconds, extracted_samples = extraction_by_seed.get(seed, (None, samples_per_seed))
        attempts.append(
            {
                "Attempt": f"Seed {seed}",
                "Inference time": inference_by_seed.get(seed),
                "Output processing time": extraction_seconds,
                "Predictions": extracted_samples or samples_per_seed,
                "Recycles": None,
            }
        )
    return attempts


def output_files(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir).resolve()
    output_dir = root / "output"
    if not output_dir.is_dir():
        return []
    return sorted(path for path in output_dir.rglob("*") if path.is_file())
