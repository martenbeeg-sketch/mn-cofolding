#!/usr/bin/env python3
"""Import the 2026-10-01 Boltz-2 capacity sweep into the MN Cofolding job browser.

Prediction files are hard-linked into normal app job folders where possible.
The original benchmark directory remains intact, while every imported attempt
gets a clickable result page and an exportable app-style job archive.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mn_cofolding.jobs import JobStore
from mn_cofolding.runtime import Settings, get_settings


EXPERIMENT = "boltz2-length-capacity-2026-10-01"
IMAGE = "mn-boltz2-opt:2.2.1-cu130"
TARGETS = (
    ("CHRNA7", "P36544"),
    ("SLC26A8", "Q96RN1"),
    ("NUP155", "O75694"),
    ("LCT", "P09848"),
    ("EP300", "Q09472"),
    ("CEP350", "Q5VT06"),
    ("DMD", "P11532"),
    ("PRKDC", "P78527"),
)
MODES = ("off", "fast", "big")
_PHASE_RE = re.compile(r"PHASE item=\S+ .*?total_s=([\d.]+)")
_EXIT_RE = re.compile(r"\brc=(\d+)")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _link_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _link_or_copy(path, target)


def _source_timestamps(path: Path, inference_seconds: float | None) -> tuple[str, str, float]:
    stat = subprocess.run(
        ["stat", "-c", "%W %Y", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    birth, modified = (int(value) for value in stat.stdout.split())
    if birth <= 0:
        modified = modified or int(path.stat().st_mtime)
        birth = modified - int(inference_seconds or 0)
    started = datetime.fromtimestamp(birth, timezone.utc)
    finished = datetime.fromtimestamp(modified, timezone.utc)
    wall = max(0.0, (finished - started).total_seconds())
    return started.isoformat(timespec="seconds"), finished.isoformat(timespec="seconds"), wall


def _a3m_from_yaml(yaml_path: Path, msa_cache_dir: Path) -> tuple[Path | None, int]:
    match = re.search(r"^\s*msa:\s*['\"]?([^'\"\s]+)", yaml_path.read_text(errors="replace"), re.MULTILINE)
    if not match:
        return None, 0
    a3m = msa_cache_dir / Path(match.group(1)).name
    if not a3m.is_file():
        return None, 0
    with a3m.open(encoding="utf-8", errors="replace") as handle:
        rows = sum(1 for line in handle if line.startswith(">"))
    return a3m, rows


def _attempt_specs(root: Path) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for gene, accession in TARGETS:
        for mode in MODES:
            attempts.append(
                {
                    "gene": gene,
                    "accession": accession,
                    "mode": mode,
                    "gpu_count": 1,
                    "profile": "full_msa",
                    "msa_cap": 8192,
                    "source_dir": root / mode / gene,
                }
            )
    for gene, accession in TARGETS:
        path = root / "big" / f"{gene}_2gpu_full"
        if path.is_dir():
            attempts.append(
                {
                    "gene": gene,
                    "accession": accession,
                    "mode": "big",
                    "gpu_count": 2,
                    "profile": "full_msa",
                    "msa_cap": 8192,
                    "source_dir": path,
                }
            )
    for gene, accession in (("LCT", "P09848"), ("EP300", "Q09472")):
        for mode in MODES:
            path = root / mode / f"{gene}_msa1024"
            if path.is_dir():
                attempts.append(
                    {
                        "gene": gene,
                        "accession": accession,
                        "mode": mode,
                        "gpu_count": 1,
                        "profile": "msa_cap_1024",
                        "msa_cap": 1024,
                        "source_dir": path,
                    }
                )
    path = root / "big" / "EP300_msa512"
    if path.is_dir():
        attempts.append(
            {
                "gene": "EP300",
                "accession": "Q09472",
                "mode": "big",
                "gpu_count": 1,
                "profile": "msa_cap_512",
                "msa_cap": 512,
                "source_dir": path,
            }
        )
    return attempts


def import_attempt(
    store: JobStore,
    settings: Settings,
    root: Path,
    spec: dict[str, Any],
) -> tuple[str, bool]:
    gene = str(spec["gene"])
    accession = str(spec["accession"])
    mode = str(spec["mode"])
    gpu_count = int(spec["gpu_count"])
    profile = str(spec["profile"])
    cap = int(spec["msa_cap"])
    source_dir = Path(spec["source_dir"])
    console_log = source_dir / "console.log"
    input_yaml = root / "inputs" / f"{gene}.yaml"
    fasta = root / "targets" / f"{gene}_{accession}.fasta"
    if not console_log.is_file() or not input_yaml.is_file() or not fasta.is_file():
        raise FileNotFoundError(f"Missing input, log, or target FASTA for {gene} at {source_dir}")

    key = f"{EXPERIMENT}:{gene}:{profile}:{mode}:{gpu_count}gpu"
    for existing in store.list(limit=10000):
        if existing.get("benchmark_key") == key:
            return str(existing["job_id"]), False

    log_text = console_log.read_text(errors="replace")
    phase_matches = list(_PHASE_RE.finditer(log_text))
    inference_seconds = float(phase_matches[-1].group(1)) if phase_matches else None
    exit_matches = list(_EXIT_RE.finditer(log_text))
    return_code = int(exit_matches[-1].group(1)) if exit_matches else 1
    prediction_root = source_dir / f"boltz_results_{gene}" / "predictions"
    structures = sorted(prediction_root.rglob("*.cif")) if prediction_root.is_dir() else []
    predicted = return_code == 0 and bool(structures)
    oom = "out of memory" in log_text.lower()
    outcome = "predicted" if predicted else "OOM" if oom else "failed"
    msa_source, msa_rows_available = _a3m_from_yaml(input_yaml, settings.msa_cache_dir or settings.reference_dir / "msa_cache")
    msa_rows_used = min(msa_rows_available, cap) if msa_rows_available else None
    started_at, finished_at, wall_seconds = _source_timestamps(console_log, inference_seconds)
    profile_label = {
        "full_msa": "Cached A3M · cap 8,192",
        "msa_cap_1024": "MSA cap 1,024",
        "msa_cap_512": "MSA cap 512",
    }[profile]
    fasta_text = fasta.read_text(encoding="utf-8")
    sequence = "".join(line.strip() for line in fasta_text.splitlines() if line.strip() and not line.startswith(">"))
    name = f"Boltz2 benchmark · {gene} ({len(sequence)} aa) · {mode} · {gpu_count} GPU · {profile_label}"
    job = store.create(
        model_name="boltz2",
        input_text=fasta_text,
        input_format="fasta",
        name=name,
        msa_mode="mmseqs2_uniref_env",
        msa_source="colabfold_server",
        seeds=[0],
        num_diffusion_samples=1,
        num_recycles=3,
        optimization_mode=mode,
        gpu=True,
    )
    job_id = str(job["metadata"]["job_id"])
    run_dir = Path(job["run_dir"])
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)

    # JobStore creates an AF3 JSON input while validating the sequence. This
    # imported Boltz run used the original YAML/A3M pair below instead.
    shutil.rmtree(run_dir / "input", ignore_errors=True)

    _link_or_copy(input_yaml, artifacts / "boltz2-input.yaml")
    _link_or_copy(fasta, artifacts / "target.fasta")
    if msa_source:
        _link_or_copy(msa_source, artifacts / "cached-input.a3m")
    _link_or_copy(console_log, run_dir / "stdout.log")
    pred_worker_log = source_dir / "pred_worker.log"
    if pred_worker_log.is_file():
        _link_or_copy(pred_worker_log, artifacts / "boltz2-opt-worker.log")
    runtime_manifest = source_dir / "mn_boltz2_opt.json"
    if runtime_manifest.is_file():
        _link_or_copy(runtime_manifest, artifacts / "mn_boltz2-opt.json")
    _link_tree(prediction_root, run_dir / "output" / f"boltz_results_{gene}" / "predictions")

    phase_label = f"MSA cap {cap:,}" if profile != "full_msa" else f"up to {cap:,} MSA rows"
    benchmark = {
        "experiment": EXPERIMENT,
        "gene": gene,
        "accession": accession,
        "sequence_length": len(sequence),
        "mode": mode,
        "gpu_count": gpu_count,
        "gpu": "NVIDIA RTX 4090",
        "profile": profile,
        "profile_label": profile_label,
        "msa_rows_available": msa_rows_available or None,
        "msa_rows_used": msa_rows_used,
        "msa_cap": cap,
        "image": IMAGE,
        "predicted": predicted,
        "outcome": outcome,
        "return_code": return_code,
        "inference_seconds": inference_seconds,
        "wall_seconds": wall_seconds,
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "source_attempt": str(source_dir),
        "structure_count": len(structures) if predicted else 0,
    }

    input_path = run_dir / "input.json"
    request = _read_json(input_path)
    request.update(
        {
            "input_file": "artifacts/boltz2-input.yaml",
            "input_format": "Boltz-2 YAML with cached A3M",
            "msa_source": "shared_cached_a3m",
            "msa_mode": f"{phase_label} ({msa_rows_used or 'unknown'} / {msa_rows_available or 'unknown'} rows)",
            "msa_rows_available": msa_rows_available,
            "msa_rows_used": msa_rows_used,
            "benchmark_key": key,
        }
    )
    _write_json(input_path, request)
    command = [
        "boltz", "predict", f"/inputs/{gene}.yaml", "--out_dir", "/work",
        "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1",
        "--max_parallel_samples", "1", "--max_msa_seqs", str(cap), "--override",
    ]
    _write_json(
        run_dir / "command.json",
        {
            "kind": "imported_benchmark",
            "image": IMAGE,
            "mode": mode,
            "gpu_count": gpu_count,
            "command": command,
            "source_attempt": str(source_dir),
        },
    )
    _write_json(artifacts / "benchmark.json", benchmark)

    store.update_status(
        job_id,
        "running",
        started_at=started_at,
        completed_at=finished_at,
        gpu_id=0,
        gpu_ids=list(range(gpu_count)),
        gpu_count=gpu_count,
        image=IMAGE,
        benchmark_key=key,
        benchmark=benchmark,
    )
    message = (
        "Imported completed Boltz-2 prediction from the capacity benchmark."
        if predicted
        else "Imported Boltz-2 capacity attempt; no prediction structure was produced (CUDA out of memory)."
        if oom
        else "Imported Boltz-2 capacity attempt; no prediction structure was produced."
    )
    result = store.finish(job_id, success=predicted, return_code=return_code, message=message)
    result["finished_at"] = finished_at
    _write_json(run_dir / "result.json", result)
    metadata_path = run_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata.update(
        {
            "created_at": started_at,
            "started_at": started_at,
            "completed_at": finished_at,
            "updated_at": finished_at,
            "benchmark_key": key,
            "benchmark": benchmark,
        }
    )
    _write_json(metadata_path, metadata)
    return job_id, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark_dir",
        nargs="?",
        type=Path,
        default=Path("/mnt/data/RESULTS/boltz2-length-capacity-20261001"),
        help="Source experiment folder (default: the recorded 2026-10-01 sweep)",
    )
    args = parser.parse_args()
    root = args.benchmark_dir.expanduser().resolve()
    settings = get_settings()
    store = JobStore(settings)
    imported = 0
    skipped = 0
    for spec in _attempt_specs(root):
        job_id, was_imported = import_attempt(store, settings, root, spec)
        imported += int(was_imported)
        skipped += int(not was_imported)
        print(f"{spec['gene']} {spec['mode']} {spec['gpu_count']}gpu {spec['profile']} -> {job_id}", flush=True)
    print(f"Imported {imported} attempt(s); reused {skipped} existing job(s).", flush=True)


if __name__ == "__main__":
    main()
