#!/usr/bin/env python3
"""Import AF3 benchmark attempts into the MN Cofolding job browser.

The benchmark runner writes a compact experiment tree, while the Streamlit app
lists only directories with app-style metadata.json. This bridges the two
layouts without rerunning predictions. Output files are hard-linked where
possible to avoid duplicating large confidence matrices and MSA-bearing input.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mn_cofolding.jobs import JobStore


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _link_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)


def _parse_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat(timespec="microseconds")


def _attempt_key(row: dict[str, Any]) -> str:
    return ":".join(
        str(row.get(key, ""))
        for key in ("gene", "mode", "num_recycles", "num_diffusion_samples", "random_seed", "image")
    )


def import_attempt(store: JobStore, benchmark_dir: Path, row: dict[str, Any]) -> str:
    input_path = benchmark_dir / "inputs" / f"{row['gene']}.json"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    sequence_path = benchmark_dir / "sequences" / f"{row['gene']}_{row['accession']}.fasta"
    if not sequence_path.is_file():
        raise FileNotFoundError(sequence_path)

    key = _attempt_key(row)
    for existing in store.list(limit=10000):
        details = existing.get("benchmark") if isinstance(existing.get("benchmark"), dict) else {}
        same_attempt = (
            details.get("gene") == row.get("gene")
            and details.get("mode") == row.get("mode")
            and details.get("profile") == f"{row.get('num_recycles')}r-{row.get('num_diffusion_samples')}s"
            and details.get("image") == row.get("image")
        )
        if existing.get("benchmark_key") == key or same_attempt:
            return str(existing["job_id"])

    mode = str(row["mode"])
    recycles = int(row["num_recycles"])
    samples = int(row["num_diffusion_samples"])
    profile = f"{recycles}r-{samples}s"
    name = f"AF3 benchmark · {row['gene']} ({row['sequence_length']} aa) · {mode} · {profile}"
    image_name = str(row.get("image", "")).split(":", 1)[0].removeprefix("mn-cofolding-af3-")
    if image_name and image_name != "opt":
        name += f" · {image_name}"
    job = store.create(
        model_name="alphafold3",
        # Use the sequence for normal app validation (the embedded benchmark
        # A3M is intentionally much larger than the UI's 2 MiB submission cap).
        input_text=sequence_path.read_text(encoding="utf-8"),
        input_format="fasta",
        name=name,
        msa_mode="mmseqs2_uniref_env",
        msa_source="colabfold_server",
        seeds=[int(row.get("random_seed", 1))],
        num_diffusion_samples=samples,
        num_recycles=recycles,
        optimization_mode=mode,
        gpu=True,
        accept_alphafold3_terms=True,
    )
    job_id = str(job["metadata"]["job_id"])
    run_dir = Path(job["run_dir"])

    # Replace the app-created copy with a hard link to the shared, exact AF3 input.
    job_input = run_dir / "input" / "input.json"
    job_input.unlink(missing_ok=True)
    try:
        os.link(input_path, job_input)
    except OSError:
        shutil.copy2(input_path, job_input)

    attempt_dir = Path(str(row["log"])).parent
    source_output = attempt_dir / "output"
    if source_output.is_dir():
        _link_tree(source_output, run_dir / "output")
    log_path = Path(str(row["log"]))
    if log_path.is_file():
        shutil.copy2(log_path, run_dir / "stdout.log")

    started_at = _parse_timestamp(row.get("started_at"))
    finished_at = _parse_timestamp(row.get("finished_at"))
    store.update_status(
        job_id,
        "running",
        started_at=started_at,
        gpu_id=int(row.get("gpu_id", 0)),
        benchmark_key=key,
    )

    completed = row.get("status") == "completed" and int(row.get("exit_code", 1)) == 0
    message = (
        "Completed benchmark prediction with official AlphaFold 3 weights."
        if completed
        else f"Benchmark attempt ended with status {row.get('status', 'failed')}; see Runner stdout for details."
    )
    result = store.finish(job_id, success=completed, return_code=int(row.get("exit_code", 1)), message=message)

    metadata_path = run_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata.update(
        {
            "name": name,
            "created_at": started_at or metadata.get("created_at"),
            "started_at": started_at,
            "completed_at": finished_at,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_id": int(row.get("gpu_id", 0)),
            "benchmark_key": key,
            "benchmark": {
                "gene": row.get("gene"),
                "accession": row.get("accession"),
                "sequence_length": row.get("sequence_length"),
                "mode": mode,
                "profile": profile,
                "image": row.get("image"),
                "gpu": row.get("gpu"),
                "gpu_id": row.get("gpu_id"),
                "wall_seconds": row.get("wall_seconds"),
                "inference_seconds": row.get("inference_seconds"),
                "gpu_memory_peak_mib": row.get("gpu_memory_peak_mib"),
                "optimization_report": row.get("optimization_report"),
            },
        }
    )
    _write_json(metadata_path, metadata)

    result["finished_at"] = finished_at or result.get("finished_at")
    _write_json(run_dir / "result.json", result)
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path, help="Directory containing inputs/ and benchmark_results*.json")
    parser.add_argument(
        "--metrics",
        action="append",
        help="Metrics JSON filename; repeat to import selected profiles (default: all benchmark_results*.json files).",
    )
    args = parser.parse_args()
    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    metric_paths = (
        [benchmark_dir / name for name in args.metrics]
        if args.metrics
        else sorted(benchmark_dir.glob("benchmark_results*.json"))
    )
    if not metric_paths:
        raise SystemExit(f"No benchmark_results*.json files found in {benchmark_dir}")

    store = JobStore()
    for metrics_path in metric_paths:
        rows = _read_json(metrics_path)
        for row in rows:
            job_id = import_attempt(store, benchmark_dir, row)
            row["app_job_id"] = job_id
            row["app_job_url"] = f"?job_id={job_id}"
            print(f"{metrics_path.name}: {row['gene']} {row['mode']} {row['status']} -> {job_id}", flush=True)
        _write_json(metrics_path, rows)


if __name__ == "__main__":
    main()
