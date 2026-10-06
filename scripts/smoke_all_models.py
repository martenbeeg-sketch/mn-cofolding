#!/usr/bin/env python3
"""Run the preview notebook's example protein through each available engine."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mn_cofolding.jobs import JobStore  # noqa: E402
from mn_cofolding.models import MODELS  # noqa: E402
from mn_cofolding.runtime import get_settings  # noqa: E402
from mn_cofolding.worker import run_job  # noqa: E402


EXAMPLE_FASTA = ">preview-notebook-example\nPIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="Comma-separated model IDs; default is every notebook model.")
    parser.add_argument("--cpu", action="store_true", help="Run without reserving a GPU (very slow for AF3-family models).")
    parser.add_argument("--accept-alphafold3-terms", action="store_true", help="Acknowledge the published terms and include official AlphaFold 3.")
    parser.add_argument("--wait-hours", type=float, default=12, help="Maximum time to wait for each queued model job.")
    parser.add_argument("--msa-mode", choices=("mmseqs2_uniref_env", "single_sequence"), default="mmseqs2_uniref_env")
    parser.add_argument(
        "--optimization-mode",
        choices=("off", "fast", "big"),
        default="off",
        help="AF3-family inference mode; AF2 accepts only off.",
    )
    parser.add_argument(
        "--msa-source",
        choices=("colabfold_server", "local_mmseqs_gpu"),
        default="colabfold_server",
        help="Generate MSAs with the shared ColabFold server or the local shared MMseqs-GPU database.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook_models = [model.name for model in MODELS]
    if args.models:
        requested = [value.strip() for value in args.models.split(",") if value.strip()]
        unknown = sorted(set(requested) - set(notebook_models))
        if unknown:
            print(f"Unknown model(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        model_names = list(dict.fromkeys(requested))
    else:
        model_names = notebook_models

    if "alphafold3" in model_names and not args.accept_alphafold3_terms:
        model_names.remove("alphafold3")
        print("SKIP alphafold3: pass --accept-alphafold3-terms after reviewing the published terms.")

    settings = get_settings()
    store = JobStore(settings)
    results: list[tuple[str, str, str]] = []
    for model_name in model_names:
        print(f"\n=== {model_name} ===", flush=True)
        try:
            job = store.create(
                model_name=model_name,
                input_text=EXAMPLE_FASTA,
                input_format="fasta",
                name=f"smoke-{model_name}",
                msa_mode=args.msa_mode,
                msa_source=args.msa_source,
                seeds=[1],
                num_diffusion_samples=1,
                num_recycles=1,
                optimization_mode=args.optimization_mode,
                cpu_threads=4,
                gpu=not args.cpu,
                accept_alphafold3_terms=args.accept_alphafold3_terms,
            )
            job_id = job["metadata"]["job_id"]
            deadline = time.monotonic() + args.wait_hours * 3600
            while time.monotonic() < deadline:
                run_job(job_id, settings=settings)
                current = store.get(job_id)
                status = current["metadata"].get("status", "unknown")
                if status in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(10)
            else:
                status = "timeout"

            detail = store.get(job_id)
            structures = [
                relative
                for relative in detail["result"].get("outputs", [])
                if Path(relative).suffix.lower() in {".cif", ".mmcif", ".pdb"}
            ]
            if status == "completed" and structures:
                print(f"PASS {model_name}: {len(structures)} structure file(s), job {job_id}", flush=True)
                results.append((model_name, "PASS", job_id))
            else:
                message = detail["result"].get("message", "")
                print(f"FAIL {model_name}: status={status}; {message}; job {job_id}", flush=True)
                print(f"  logs: {detail['run_dir'] / 'stdout.log'} and {detail['run_dir'] / 'stderr.log'}", flush=True)
                results.append((model_name, "FAIL", job_id))
        except Exception as exc:  # Keep going so one bad engine does not hide the rest.
            print(f"FAIL {model_name}: {type(exc).__name__}: {exc}", flush=True)
            results.append((model_name, "FAIL", "not-created"))

    print("\nModel smoke summary")
    for model_name, status, job_id in results:
        print(f"{status:4}  {model_name:22} {job_id}")
    failures = sum(status != "PASS" for _name, status, _job in results)
    print(f"{len(results) - failures}/{len(results)} model run(s) passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
