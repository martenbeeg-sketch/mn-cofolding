from __future__ import annotations

import json
import os
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .inputs import validate_input, validate_model_input_compatibility
from .models import get_model
from .runtime import Settings, get_settings


ACTIVE_STATUSES = {"queued", "preparing", "running", "cancelling"}
FINAL_STATUSES = {"completed", "failed", "cancelled"}
VALID_STATUSES = ACTIVE_STATUSES | FINAL_STATUSES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _job_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"


class JobStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def ensure(self) -> None:
        self.settings.ensure_dirs()

    def job_dir(self, job_id: str) -> Path:
        if not job_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in job_id):
            raise ValueError("Invalid job ID")
        path = (self.settings.runs_dir / "cofolding" / job_id).resolve()
        if self.settings.runs_dir not in path.parents:
            raise ValueError("Invalid job ID")
        return path

    def create(
        self,
        *,
        model_name: str,
        input_text: str,
        input_format: str = "fasta",
        name: str | None = None,
        msa_mode: str = "mmseqs2_uniref_env",
        msa_source: str = "colabfold_server",
        seeds: list[int] | None = None,
        num_diffusion_samples: int = 5,
        num_recycles: int | None = 10,
        optimization_mode: str = "off",
        msa_max_depth: int = 1024,
        cpu_threads: int = 4,
        gpu: bool = True,
        accept_alphafold3_terms: bool = False,
    ) -> dict:
        self.ensure()
        model = get_model(model_name)
        if optimization_mode not in {"off", "exact", "fast", "big"}:
            raise ValueError("optimization_mode must be one of: off, exact, fast, big")
        if optimization_mode == "exact" and model.name not in {"alphafold3", "esmfold2", "chai1"}:
            raise ValueError("Exact mode is currently available for official AlphaFold 3, ESMFold2, and Chai-1 kit weights only")
        if model.family == "af2" and optimization_mode != "off":
            raise ValueError("Optimization modes currently apply to AF3-family models only")
        if model.name in {"esmfold2", "chai1"} and not gpu:
            raise ValueError(f"{model.label} requires a reserved GPU")
        if optimization_mode != "off" and not gpu:
            raise ValueError("AF3 optimization modes require a reserved GPU")
        if msa_mode not in {"mmseqs2_uniref_env", "mmseqs2_uniref_env_envpair", "mmseqs2_uniref", "single_sequence"}:
            raise ValueError(f"Unsupported MSA mode: {msa_mode}")
        if msa_source not in {"colabfold_server", "local_mmseqs_gpu", "single_sequence"}:
            raise ValueError(f"Unsupported MSA source: {msa_source}")
        if model.name in {"esmfold2", "chai1"} and msa_source not in {"local_mmseqs_gpu", "single_sequence"}:
            raise ValueError(f"{model.label} supports local MMseqs-GPU MSAs or single-sequence mode")
        if model.name in {"esmfold2", "chai1"} and (msa_source == "single_sequence") != (msa_mode == "single_sequence"):
            raise ValueError(f"{model.label} MSA source and mode must both select local MMseqs-GPU or single-sequence")
        if msa_source == "local_mmseqs_gpu" and (msa_mode == "single_sequence" or not gpu):
            raise ValueError("Local MMseqs-GPU needs MSA enabled and a reserved GPU")
        if num_diffusion_samples not in {1, 2, 3, 4, 5}:
            raise ValueError("num_diffusion_samples must be from 1 through 5")
        if num_recycles is not None and not 1 <= int(num_recycles) <= 100:
            raise ValueError("num_recycles must be from 1 through 100")
        if not 1 <= int(msa_max_depth) <= 8192:
            raise ValueError("msa_max_depth must be from 1 through 8192")
        if not 1 <= int(cpu_threads) <= 256:
            raise ValueError("cpu_threads must be from 1 through 256")
        if model_name == "alphafold3" and not accept_alphafold3_terms:
            raise ValueError("Accept AlphaFold 3's published weight terms before using the official model")
        if seeds and (not all(0 <= int(seed) <= 2**31 - 1 for seed in seeds) or len(seeds) > 32):
            raise ValueError("Provide up to 32 non-negative integer seeds")
        validated = validate_input(input_text, input_format, model_name)
        validate_model_input_compatibility(model_name, validated.input_format)
        if model.name in {"esmfold2", "chai1"}:
            if validated.input_format != "alphafold3_json":
                raise ValueError(f"{model.label} accepts protein chains from the molecule-fields input")
            import json

            payload = json.loads(validated.text)
            if any(not isinstance(entity, dict) or not isinstance(entity.get("protein"), dict) for entity in payload.get("sequences", [])):
                raise ValueError(f"{model.label} accepts protein chains only; remove DNA, RNA, and ligand entities")
        if msa_source == "local_mmseqs_gpu" and model.family == "af2" and validated.chain_count != 1:
            raise ValueError("Local MMseqs-GPU input for AlphaFold 2 currently supports one protein chain per job")
        seed_values = list(dict.fromkeys(int(seed) for seed in (seeds or [1])))
        if model.family == "af2" and seed_values != list(range(seed_values[0], seed_values[0] + len(seed_values))):
            raise ValueError("AlphaFold 2 accepts consecutive seed values only")
        if model.family == "af3" and validated.input_format == "fasta":
            from .inputs import fasta_to_af3_json

            validated = fasta_to_af3_json(
                validated,
                name=name or "cofold",
                seeds=seed_values,
                msa_mode=msa_mode,
            )
        elif model.family == "af3" and validated.input_format == "alphafold3_json":
            import json

            input_payload = json.loads(validated.text)
            input_payload["modelSeeds"] = seed_values
            validated = validated.__class__(
                json.dumps(input_payload, indent=2) + "\n",
                "input.json",
                "alphafold3_json",
                validated.chain_count,
                validated.residue_count,
            )
        if not gpu:
            # CPU inference is intentionally explicit because AF3 cofolding is very slow without a GPU.
            pass
        job_id = _job_id()
        gpu_count = 2 if gpu and model.name in {"alphafold3", "esmfold2"} and optimization_mode == "big" else (1 if gpu else 0)
        run_dir = self.job_dir(job_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "artifacts").mkdir()
        (run_dir / "input").mkdir()
        (run_dir / "output").mkdir()
        (run_dir / "input" / validated.file_name).write_text(validated.text, encoding="utf-8")
        for log_name in ("stdout.log", "stderr.log"):
            (run_dir / log_name).write_text("", encoding="utf-8")
        created = utc_now()
        clean_name = (name or f"{model.label} {job_id[-5:]}").strip()[:120]
        payload = {
            "job_id": job_id,
            "name": clean_name,
            "model": model_name,
            "input_file": f"input/{validated.file_name}",
            "input_format": validated.input_format,
            "chain_count": validated.chain_count,
            "residue_count": validated.residue_count,
            "msa_mode": msa_mode,
            "msa_source": msa_source,
            "seeds": seed_values,
            "num_diffusion_samples": int(num_diffusion_samples),
            "num_recycles": int(num_recycles) if num_recycles is not None else None,
            "optimization_mode": optimization_mode,
            "msa_max_depth": int(msa_max_depth),
            "cpu_threads": int(cpu_threads),
            "gpu": bool(gpu),
            "gpu_count": gpu_count,
            "accept_alphafold3_terms": bool(accept_alphafold3_terms),
        }
        if model.family == "af2" and msa_mode != "single_sequence":
            from .msa_cache import af2_msa_key

            payload["msa_cache_key"] = af2_msa_key(input_text, msa_mode)
        metadata = {
            "job_id": job_id,
            "name": clean_name,
            "model": model_name,
            "family": model.family,
            "status": "queued",
            "gpu_count": gpu_count,
            "created_at": created,
            "updated_at": created,
            "host": socket.gethostname(),
            "schema_version": 1,
        }
        _write_json(run_dir / "input.json", payload)
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "worker_request.json", {"kind": "cofold", "requested_at": created})
        _write_json(run_dir / "command.json", {"mode": "docker", "command": []})
        return self.get(job_id)

    def get(self, job_id: str) -> dict:
        run_dir = self.job_dir(job_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(job_id)
        return {
            "metadata": _read_json(run_dir / "metadata.json"),
            "input": _read_json(run_dir / "input.json"),
            "result": _read_json(run_dir / "result.json"),
            "run_dir": run_dir,
        }

    def list(self, *, limit: int = 100) -> list[dict]:
        root = self.settings.runs_dir / "cofolding"
        if not root.is_dir():
            return []
        items = []
        for path in root.iterdir():
            if path.is_dir() and (path / "metadata.json").is_file():
                metadata = _read_json(path / "metadata.json")
                metadata.setdefault("job_id", path.name)
                items.append(metadata)
        return sorted(items, key=lambda item: str(item.get("created_at", "")), reverse=True)[: max(1, int(limit))]

    def update_status(self, job_id: str, status: str, **fields: object) -> dict:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        run_dir = self.job_dir(job_id)
        metadata_path = run_dir / "metadata.json"
        metadata = _read_json(metadata_path)
        if not metadata:
            raise FileNotFoundError(job_id)
        metadata.update(fields)
        metadata["status"] = status
        metadata["updated_at"] = utc_now()
        if status in FINAL_STATUSES:
            metadata.setdefault("completed_at", metadata["updated_at"])
        _write_json(metadata_path, metadata)
        return metadata

    def finish(self, job_id: str, *, success: bool, return_code: int, message: str = "") -> dict:
        run_dir = self.job_dir(job_id)
        metadata = _read_json(run_dir / "metadata.json")
        if metadata.get("cancel_requested"):
            status = "cancelled"
        else:
            status = "completed" if success else "failed"
        outputs = []
        output_dir = run_dir / "output"
        if output_dir.is_dir():
            outputs = [
                str(path.relative_to(run_dir))
                for path in sorted(output_dir.rglob("*"))
                if path.is_file()
            ]
        result = {
            "success": status == "completed",
            "status": status,
            "return_code": int(return_code),
            "message": message,
            "outputs": outputs,
            "finished_at": utc_now(),
        }
        _write_json(run_dir / "result.json", result)
        self.update_status(job_id, status, return_code=int(return_code), output_files=len(outputs))
        return result

    def cancel(self, job_id: str) -> dict:
        current = self.get(job_id)
        status = current["metadata"].get("status")
        if status == "queued":
            self.update_status(job_id, "cancelled", cancel_requested=True, cancel_reason="Cancelled before dispatch")
            _write_json(self.job_dir(job_id) / "result.json", {"success": False, "status": "cancelled", "message": "Cancelled before dispatch", "outputs": []})
        elif status in {"preparing", "running", "cancelling"}:
            self.update_status(job_id, "cancelling", cancel_requested=True, cancel_requested_at=utc_now())
        else:
            raise ValueError(f"Job {job_id} is already {status}")
        return self.get(job_id)

    def artifact_path(self, job_id: str, relative_path: str) -> Path:
        base = self.job_dir(job_id).resolve()
        candidate = (base / relative_path).resolve()
        if base not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(relative_path)
        return candidate


def job_store(settings: Settings | None = None) -> JobStore:
    return JobStore(settings)
