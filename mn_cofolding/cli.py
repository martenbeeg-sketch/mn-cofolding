from __future__ import annotations

from pathlib import Path

import typer
import subprocess
import sys

from .jobs import JobStore
from .models import MODELS
from .portability import export_workdir, import_archive, verify_archive
from .runtime import get_settings
from .worker import ensure_worker_service, worker_service


app = typer.Typer(help="Portable Streamlit cofolding workbench", no_args_is_help=True)


@app.command()
def init():
    """Create configured app, run, and reference directories."""
    settings = get_settings()
    settings.ensure_dirs()
    typer.echo(f"app home:   {settings.app_home}")
    typer.echo(f"runs:       {settings.runs_dir}")
    typer.echo(f"references: {settings.reference_dir}")


@app.command("app")
def app_ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8501, "--port", min=1, max=65535),
):
    """Run the Streamlit interface."""
    app_path = Path(__file__).with_name("app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.address", host, "--server.port", str(port)]
    raise SystemExit(subprocess.call(command))


@app.command()
def api(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
):
    """Run the local REST API."""
    import uvicorn

    uvicorn.run("mn_cofolding.api:app", host=host, port=port, reload=False)


@app.command()
def worker(
    once: bool = typer.Option(False, "--once", help="Dispatch at most one queued job, then exit."),
    poll_seconds: float = typer.Option(2.0, "--poll-seconds", min=0.25),
):
    """Run or start the background local worker."""
    worker_service(once=once, poll_seconds=poll_seconds)


@app.command("worker-start")
def worker_start():
    """Start the detached queue worker service."""
    ensure_worker_service()
    typer.echo("Worker service is running.")


@app.command()
def submit(
    input_file: Path = typer.Argument(..., exists=True, readable=True),
    model: str = typer.Option("openfold3", "--model"),
    name: str | None = typer.Option(None, "--name"),
    msa_mode: str = typer.Option("mmseqs2_uniref_env", "--msa-mode"),
    msa_source: str = typer.Option("colabfold_server", "--msa-source", help="colabfold_server or local_mmseqs_gpu"),
    num_diffusion_samples: int = typer.Option(5, "--num-samples", min=1, max=5),
    num_recycles: int = typer.Option(10, "--num-recycles", min=1, max=100),
    optimization_mode: str = typer.Option("off", "--optimization-mode", help="AF3 mode: off, exact, fast, or big"),
    gpu: bool = typer.Option(True, "--gpu/--cpu"),
    accept_alphafold3_terms: bool = typer.Option(False, "--accept-alphafold3-terms"),
):
    """Queue an input FASTA or AlphaFold 3 JSON file."""
    input_format = "alphafold3_json" if input_file.suffix.lower() == ".json" else "fasta"
    job = JobStore().create(
        model_name=model,
        input_text=input_file.read_text(encoding="utf-8"),
        input_format=input_format,
        name=name or input_file.stem,
        msa_mode=msa_mode,
        msa_source=msa_source,
        num_diffusion_samples=num_diffusion_samples,
        num_recycles=num_recycles,
        optimization_mode=optimization_mode,
        gpu=gpu,
        accept_alphafold3_terms=accept_alphafold3_terms,
    )
    ensure_worker_service()
    typer.echo(job["metadata"]["job_id"])


@app.command("jobs")
def list_jobs(limit: int = typer.Option(100, min=1, max=500)):
    """List recent local jobs."""
    for item in JobStore().list(limit=limit):
        typer.echo(f"{item.get('job_id')}\t{item.get('status')}\t{item.get('model')}\t{item.get('name')}")


@app.command()
def status(job_id: str):
    """Show one job's metadata and result record."""
    import json

    typer.echo(json.dumps(JobStore().get(job_id), default=str, indent=2))


@app.command()
def cancel(job_id: str):
    """Cancel a queued or running job."""
    try:
        item = JobStore().cancel(job_id)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(item["metadata"].get("status"))


@app.command()
def models():
    """Show the supported preview and AF2 model names."""
    for item in MODELS:
        typer.echo(f"{item.name}\t{item.family}\t{item.label}")


@app.command()
def export(output: Path = typer.Argument(..., help="Output ZIP file for completed run data.")):
    """Export completed job records and outputs for transfer to another computer."""
    try:
        result = export_workdir(output)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(result)


@app.command()
def verify(archive: Path = typer.Argument(..., exists=True, readable=True)):
    """Verify a portability archive and its checksums."""
    import json

    typer.echo(json.dumps(verify_archive(archive), indent=2))


@app.command("import")
def import_runs(
    archive: Path = typer.Argument(..., exists=True, readable=True),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Import completed runs into the configured work directory."""
    try:
        result = import_archive(archive, overwrite=overwrite)
    except (ValueError, RuntimeError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Imported {result.get('job_count', 0)} job(s).")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
