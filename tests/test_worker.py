from __future__ import annotations

from pathlib import Path

import pytest

from mn_cofolding.jobs import JobStore
from mn_cofolding.msa_cache import af2_msa_path, cache_af2_msa
from mn_cofolding.worker import portable_command, run_job


EXAMPLE_FASTA = ">test\nPIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK\n"


class DummyLease:
    def heartbeat(self):
        pass

    def set_owner_pid(self, _pid):
        pass

    def release(self):
        return True


class FakeProcess:
    pid = 12345
    returncode = 0

    def __init__(self, command, *, cwd, **_kwargs):
        output = Path(cwd) / "output" / "prediction.cif"
        output.write_text("data_prediction\n")

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


@pytest.fixture
def patched_worker(monkeypatch):
    monkeypatch.setattr("mn_cofolding.worker.FileLease.acquire", classmethod(lambda _cls, *_args, **_kwargs: DummyLease()))
    lease = DummyLease()
    monkeypatch.setattr("mn_cofolding.worker._acquire_resources", lambda *_args, **_kwargs: (lease, 0, [lease]))
    monkeypatch.setattr("mn_cofolding.worker.docker_command", lambda *_args, **_kwargs: ["docker", "run"])


def test_worker_marks_job_complete_when_structure_is_written(settings, patched_worker):
    store = JobStore(settings)
    job = store.create(model_name="openfold3", input_text=EXAMPLE_FASTA)
    job_id = job["metadata"]["job_id"]
    assert run_job(job_id, settings=settings, popen_factory=FakeProcess)
    detail = store.get(job_id)
    assert detail["metadata"]["status"] == "completed"
    assert detail["metadata"]["started_at"]
    assert detail["result"]["outputs"] == ["output/prediction.cif"]


def test_portable_command_keeps_nested_cache_as_its_own_path_role(settings, tmp_path):
    run_dir = tmp_path / "runs" / "cofolding" / "job-1"
    command = [
        "--mount",
        f"type=bind,src={settings.reference_dir},dst=/reference",
        "--mount",
        f"type=bind,src={settings.cache_dir},dst=/cache",
        "--mount",
        f"type=bind,src={run_dir},dst=/work",
    ]
    portable = portable_command(command, run_dir, settings)
    assert "type=bind,src=${REFERENCE_DIR},dst=/reference" in portable
    assert "type=bind,src=${CACHE_DIR},dst=/cache" in portable
    assert "type=bind,src=${RUN_DIR},dst=/work" in portable


def test_worker_fails_if_runner_does_not_write_a_structure(settings, patched_worker):
    class NoOutputProcess(FakeProcess):
        def __init__(self, command, *, cwd, **kwargs):
            self.cwd = Path(cwd)

    store = JobStore(settings)
    job = store.create(model_name="openfold3", input_text=EXAMPLE_FASTA)
    job_id = job["metadata"]["job_id"]
    assert not run_job(job_id, settings=settings, popen_factory=NoOutputProcess)
    detail = store.get(job_id)
    assert detail["metadata"]["status"] == "failed"
    assert "without writing a structure" in detail["result"]["message"]


def test_af2_msa_result_is_saved_for_later_jobs(settings):
    store = JobStore(settings)
    job = store.create(model_name="af2_ptm", input_text=EXAMPLE_FASTA)
    request = job["input"]
    produced = job["run_dir"] / "output" / "prediction.a3m"
    produced.write_text(">query\nPIAQIHILEG\n")

    saved = cache_af2_msa(job["run_dir"], settings)

    assert saved == af2_msa_path(settings, request["msa_mode"], request["msa_cache_key"])
    assert saved.read_text() == produced.read_text()
