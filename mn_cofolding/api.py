from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .jobs import JobStore
from .models import MODELS
from .references import all_reference_statuses
from .runtime import Settings, get_settings
from .worker import ensure_worker_service


class CreateJob(BaseModel):
    model: str
    input_text: str = Field(min_length=1, max_length=2_097_152)
    input_format: Literal["fasta", "alphafold3_json"] = "fasta"
    name: str | None = Field(default=None, max_length=120)
    msa_mode: Literal["mmseqs2_uniref_env", "mmseqs2_uniref_env_envpair", "mmseqs2_uniref", "single_sequence"] = "mmseqs2_uniref_env"
    msa_source: Literal["colabfold_server", "local_mmseqs_gpu"] = "colabfold_server"
    seeds: list[int] = Field(default_factory=lambda: [1], min_length=1, max_length=32)
    num_diffusion_samples: int = Field(default=5, ge=1, le=5)
    num_recycles: int | None = Field(default=10, ge=1, le=100)
    optimization_mode: Literal["off", "exact", "fast", "big"] = "off"
    cpu_threads: int = Field(default=4, ge=1, le=256)
    gpu: bool = True
    accept_alphafold3_terms: bool = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = JobStore(settings)
    app = FastAPI(title="mn-cofolding API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "app_home": str(settings.app_home), "runs_dir": str(settings.runs_dir)}

    @app.get("/models")
    def models():
        return [{"name": model.name, "label": model.label, "family": model.family, "description": model.description} for model in MODELS]

    @app.get("/references")
    def references():
        return [item.__dict__ for item in all_reference_statuses(settings)]

    @app.post("/jobs", status_code=201)
    def submit(request: CreateJob):
        try:
            job = store.create(model_name=request.model, **request.model_dump(exclude={"model"}))
            ensure_worker_service(settings)
            return _public_job(job)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/jobs")
    def list_jobs(limit: int = 100):
        return store.list(limit=min(max(limit, 1), 500))

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return _public_job(store.get(job_id))
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return _public_job(store.cancel(job_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/jobs/{job_id}/files/{relative_path:path}")
    def get_file(job_id: str, relative_path: str):
        try:
            path = store.artifact_path(job_id, relative_path)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        return FileResponse(path, filename=Path(relative_path).name)

    return app


def _public_job(job: dict) -> dict:
    return {key: value for key, value in job.items() if key != "run_dir"}


app = create_app()
