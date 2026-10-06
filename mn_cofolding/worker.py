from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from uuid import uuid4

from mn_compute_scheduler import (
    FileLease,
    acquire_cpu_lease,
    acquire_gpu_lease,
    cpu_pool_capacity,
    discover_gpu_ids,
    shared_state_dir,
)

from .engine import docker_command
from .chai1 import prepare_chai1_input
from .esmfold2 import prepare_esmfold2_input
from .jobs import JobStore, _read_json, _write_json, utc_now
from .local_msa import prepare_local_msa_inputs
from .models import get_model, image_for_model
from .msa_cache import cache_af2_msa
from .runtime import Settings, get_settings


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"


def _append(path: Path, chunk: str) -> None:
    if not chunk:
        return
    with path.open("a", encoding="utf-8", errors="replace") as stream:
        stream.write(chunk)
        stream.flush()


def portable_command(command: list[str], run_dir: Path, settings: Settings) -> list[str]:
    """Replace host-specific bind paths with stable path-role placeholders."""
    replacements = (
        (str(settings.cache_dir.resolve()), "${CACHE_DIR}"),
        (str(settings.reference_dir.resolve()), "${REFERENCE_DIR}"),
        (str(run_dir.resolve()), "${RUN_DIR}"),
    )
    return [_replace_all(value, replacements) for value in command]


def _replace_all(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for source, placeholder in replacements:
        value = value.replace(source, placeholder)
    return value


def _heartbeat_loop(leases: list, stop: threading.Event) -> None:
    while not stop.wait(20):
        for lease in leases:
            try:
                lease.heartbeat()
            except Exception:
                # Job execution continues; stale lease recovery remains the scheduler's job.
                pass


def _acquire_resources(store: JobStore, job_id: str, worker_id: str, settings: Settings):
    payload = _read_json(store.job_dir(job_id) / "input.json")
    state_root = settings.scheduler_state_dir or shared_state_dir()
    capacity = cpu_pool_capacity()
    requested = int(payload.get("cpu_threads", 4))
    if requested > capacity:
        raise ValueError(f"Job requests {requested} CPU slots, but the shared scheduler capacity is {capacity}")
    cpu_lease = acquire_cpu_lease(
        requested,
        run_id=job_id,
        worker_id=worker_id,
        workflow="cofolding",
        app_id="mn-cofolding",
        state_root=state_root,
        capacity=capacity,
    )
    if cpu_lease is None:
        return None
    leases: list = [cpu_lease]
    gpu_selection = None
    if payload.get("gpu"):
        gpu_ids = discover_gpu_ids()
        requested_gpu_count = int(payload.get("gpu_count", 1))
        if requested_gpu_count == 2 and (
            str(payload.get("model", "")) not in {"alphafold3", "esmfold2"}
            or str(payload.get("optimization_mode", "off")) != "big"
        ):
            cpu_lease.release()
            raise ValueError("Only official AlphaFold 3 or ESMFold2 big mode can reserve two GPUs")
        # Folding uses GPU 0. Official AF3 and ESMFold2 big are the only modes allowed to
        # reserve a second device, and it must hold both leases together.
        required_gpu_ids = [0, 1] if requested_gpu_count == 2 else [0]
        if requested_gpu_count not in (1, 2) or any(gpu_id not in gpu_ids for gpu_id in required_gpu_ids):
            cpu_lease.release()
            return None
        gpu_claims = []
        for gpu_id in required_gpu_ids:
            gpu_lease = acquire_gpu_lease(
                gpu_id,
                run_id=job_id,
                worker_id=worker_id,
                workflow="cofolding",
                app_id="mn-cofolding",
                state_root=state_root,
            )
            if gpu_lease is None:
                for claimed in gpu_claims:
                    claimed.release()
                cpu_lease.release()
                return None
            gpu_claims.append(gpu_lease)
        leases.extend(gpu_claims)
        gpu_selection = required_gpu_ids[0] if requested_gpu_count == 1 else required_gpu_ids
    return cpu_lease, gpu_selection, leases


def run_job(job_id: str, *, settings: Settings | None = None, popen_factory=subprocess.Popen) -> bool:
    settings = settings or get_settings()
    settings.ensure_dirs()
    store = JobStore(settings)
    run_dir = store.job_dir(job_id)
    metadata = _read_json(run_dir / "metadata.json")
    if metadata.get("status") not in {"queued", "preparing"}:
        return False
    owner = _worker_id()
    claim = FileLease.acquire(
        run_dir / ".worker-claim.json",
        run_id=job_id,
        owner_id=owner,
        stale_after_seconds=3600,
    )
    if claim is None:
        return False
    leases = []
    process = None
    heartbeat_stop = threading.Event()
    heartbeat_thread = None
    try:
        store.update_status(
            job_id,
            "preparing",
            worker_id=owner,
            worker_pid=os.getpid(),
            worker_started_at=utc_now(),
        )
        acquired = _acquire_resources(store, job_id, owner, settings)
        if acquired is None:
            store.update_status(job_id, "queued", scheduler_wait_reason="Waiting for a shared CPU/GPU resource lease")
            return False
        cpu_lease, gpu_selection, leases = acquired
        if isinstance(gpu_selection, (list, tuple)):
            gpu_ids = [int(value) for value in gpu_selection]
        elif gpu_selection is None:
            gpu_ids = []
        else:
            gpu_ids = [int(gpu_selection)]
        gpu_id = gpu_ids[0] if gpu_ids else None
        request = _read_json(run_dir / "input.json")
        if request.get("msa_source") == "local_mmseqs_gpu":
            store.update_status(job_id, "preparing", msa_preparation_started_at=utc_now())
            request.update(prepare_local_msa_inputs(run_dir, request, gpu_id=gpu_id, settings=settings))
            store.update_status(job_id, "preparing", msa_preparation_finished_at=utc_now())
            _write_json(run_dir / "input.json", request)
        if request.get("model") == "esmfold2":
            request.update(prepare_esmfold2_input(run_dir, request))
            _write_json(run_dir / "input.json", request)
        if request.get("model") == "chai1":
            request.update(prepare_chai1_input(run_dir, request))
            _write_json(run_dir / "input.json", request)
        command = docker_command(run_dir, gpu_id=gpu_id, gpu_ids=gpu_ids, settings=settings)
        request_model = get_model(str(request["model"]))
        image = image_for_model(
            request_model.name,
            af3_image=settings.af3_image,
            af2_image=settings.af2_image,
            esmfold2_image=settings.esmfold2_image,
            chai1_image=settings.chai1_image,
        )
        portable = portable_command(command, run_dir, settings)
        _write_json(run_dir / "command.json", {"mode": "docker", "command": portable, "image": image})
        store.update_status(
            job_id,
            "running",
            worker_id=owner,
            gpu_id=gpu_id,
            gpu_ids=gpu_ids,
            gpu_count=len(gpu_ids),
            scheduler_wait_reason="",
            started_at=utc_now(),
        )
        heartbeat_thread = threading.Thread(target=_heartbeat_loop, args=(leases, heartbeat_stop), daemon=True)
        heartbeat_thread.start()
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        with stdout_path.open("a", encoding="utf-8", errors="replace") as stdout_stream, stderr_path.open("a", encoding="utf-8", errors="replace") as stderr_stream:
            process = popen_factory(
                command,
                cwd=run_dir,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
                start_new_session=True,
            )
            for lease in leases:
                try:
                    lease.set_owner_pid(process.pid)
                except Exception:
                    pass
            while process.poll() is None:
                time.sleep(1)
                status = _read_json(run_dir / "metadata.json").get("status")
                if status == "cancelling":
                    stop_cmd = [settings.docker_executable, "stop", f"mn-cofolding-{job_id}"]
                    subprocess.run(stop_cmd, capture_output=True, text=True, timeout=60, check=False)
                    break
            return_code = process.wait()
        if get_model(str(request.get("model"))).family == "af2":
            try:
                cache_af2_msa(run_dir, settings)
            except Exception as cache_error:
                _append(run_dir / "stderr.log", f"\nCould not save reusable AF2 MSA: {cache_error}\n")
        cancel_requested = bool(_read_json(run_dir / "metadata.json").get("cancel_requested"))
        output_dir = run_dir / "output"
        structure_found = any(path.suffix.lower() in {".cif", ".mmcif", ".pdb"} for path in output_dir.rglob("*") if path.is_file())
        message = "" if structure_found else "Runner exited without writing a structure file"
        result = store.finish(
            job_id,
            success=return_code == 0 and not cancel_requested and structure_found,
            return_code=return_code,
            message=message,
        )
        return result["status"] == "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _append(run_dir / "stderr.log", "\n" + traceback.format_exc() + "\n")
        store.finish(job_id, success=False, return_code=int(getattr(process, "returncode", -1) or -1), message=error)
        return False
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
        for lease in leases:
            try:
                lease.release()
            except Exception:
                pass
        claim.release()


def dispatch_once(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    store = JobStore(settings)
    for metadata in store.list(limit=500):
        if metadata.get("status") == "queued":
            run_job(str(metadata["job_id"]), settings=settings)
            return True
    return False


def worker_service(*, poll_seconds: float = 2.0, once: bool = False, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    settings.ensure_dirs()
    if once:
        dispatch_once(settings)
        return
    while True:
        handled = dispatch_once(settings)
        time.sleep(0.25 if handled else max(0.25, poll_seconds))


def ensure_worker_service(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    settings.ensure_dirs()
    service_dir = settings.app_home / "worker-service"
    service_dir.mkdir(parents=True, exist_ok=True)
    lock = service_dir / "service.json"
    existing = _read_json(lock)
    pid = int(existing.get("pid") or 0)
    if pid:
        try:
            os.kill(pid, 0)
            return
        except (ProcessLookupError, PermissionError):
            pass
    log_path = service_dir / "worker.log"
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [sys.executable, "-m", "mn_cofolding.worker", "--service"],
            cwd=Path(__file__).resolve().parents[1],
            stdout=stream,
            stderr=stream,
            start_new_session=True,
            close_fds=True,
        )
    _write_json(lock, {"pid": process.pid, "started_at": time.time()})


def main() -> None:
    parser = argparse.ArgumentParser(description="mn-cofolding queued job worker")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--service", action="store_true", help="Run the persistent local worker service")
    mode.add_argument("--run-job", help="Run a single job by ID")
    mode.add_argument("--once", action="store_true", help="Dispatch at most one queued job and exit")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.run_job:
        run_job(args.run_job)
    else:
        worker_service(poll_seconds=args.poll_seconds, once=args.once)


if __name__ == "__main__":
    main()
