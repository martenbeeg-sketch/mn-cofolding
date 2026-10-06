from __future__ import annotations

import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from mn_cofolding.api import create_app
from mn_cofolding.jobs import JobStore
from mn_cofolding.portability import (
    PortabilityError,
    export_job_archive,
    export_workdir,
    import_archive,
    verify_archive,
)


EXAMPLE_FASTA = ">test\nPIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK\n"


def test_api_health_and_create_job(settings, monkeypatch):
    monkeypatch.setattr("mn_cofolding.api.ensure_worker_service", lambda _settings: None)
    client = TestClient(create_app(settings))
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/jobs", json={"model": "af2_ptm", "input_text": EXAMPLE_FASTA})
    assert response.status_code == 201
    job_id = response.json()["metadata"]["job_id"]
    assert response.json()["input"]["num_recycles"] == 10
    assert response.json()["input"]["num_diffusion_samples"] == 5
    assert response.json()["input"]["optimization_mode"] == "off"
    assert client.get(f"/jobs/{job_id}").status_code == 200
    assert client.get("/models").status_code == 200


def test_api_accepts_big_mode_for_af3_and_rejects_it_for_af2(settings, monkeypatch):
    monkeypatch.setattr("mn_cofolding.api.ensure_worker_service", lambda _settings: None)
    client = TestClient(create_app(settings))
    response = client.post("/jobs", json={"model": "openfold3", "input_text": EXAMPLE_FASTA, "optimization_mode": "big"})
    assert response.status_code == 201
    assert response.json()["input"]["optimization_mode"] == "big"
    af2_response = client.post("/jobs", json={"model": "af2_ptm", "input_text": EXAMPLE_FASTA, "optimization_mode": "fast"})
    assert af2_response.status_code == 422


def test_export_verify_import_moves_completed_job_and_result(settings, tmp_path):
    store = JobStore(settings)
    job = store.create(model_name="af2_ptm", input_text=EXAMPLE_FASTA)
    job_id = job["metadata"]["job_id"]
    run_dir = job["run_dir"]
    (run_dir / "output" / "prediction.cif").write_text("data_prediction\n")
    store.update_status(job_id, "completed")
    archive = export_workdir(tmp_path / "runs.zip", settings)
    manifest = verify_archive(archive)
    assert manifest["job_count"] == 1

    target = tmp_path / "second-computer"
    from mn_cofolding.runtime import Settings

    target_settings = Settings(
        app_home=target / "app",
        runs_dir=target / "results" / "runs",
        reference_dir=target / "reference_files",
        cache_dir=target / "reference_files" / "mn-cofolding" / "cache",
        msa_cache_dir=target / "reference_files" / "msa_cache",
        temp_dir=target / "tmp",
        af3_image="mn-cofolding-af3:test",
        af2_image="mn-colabfold:test",
        docker_executable="docker",
    )
    imported = import_archive(archive, target_settings)
    assert imported["job_count"] == 1
    assert (target_settings.runs_dir / "cofolding" / job_id / "output" / "prediction.cif").read_text() == "data_prediction\n"


def test_verify_rejects_tampered_archive(settings, tmp_path):
    store = JobStore(settings)
    job = store.create(model_name="af2_ptm", input_text=EXAMPLE_FASTA)
    store.update_status(job["metadata"]["job_id"], "completed")
    archive = export_workdir(tmp_path / "clean.zip", settings)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(tampered, "w") as output:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename.endswith("/input.json"):
                data += b"tamper"
            output.writestr(item, data)
    with pytest.raises(PortabilityError, match="Checksum|contents"):
        verify_archive(tampered)


def test_job_result_zip_contains_input_output_logs_and_manifest(settings):
    store = JobStore(settings)
    job = store.create(model_name="openfold3", input_text=EXAMPLE_FASTA)
    job_id = job["metadata"]["job_id"]
    run_dir = job["run_dir"]
    (run_dir / "output" / "prediction.cif").write_text("data_prediction\n")
    (run_dir / "stdout.log").write_text("fold completed\n")
    store.update_status(job_id, "completed")

    archive_bytes = export_job_archive(job_id, settings)

    from io import BytesIO

    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        assert f"{job_id}/{job['input']['input_file']}" in names
        assert f"{job_id}/output/prediction.cif" in names
        assert f"{job_id}/stdout.log" in names
        manifest = json.loads(archive.read(f"{job_id}/job-export.json"))
        assert manifest["job_id"] == job_id
        assert manifest["application"] == "mn-cofolding-job"


def test_job_result_zip_waits_for_active_job(settings):
    job = JobStore(settings).create(model_name="openfold3", input_text=EXAMPLE_FASTA)
    with pytest.raises(PortabilityError, match="finish"):
        export_job_archive(job["metadata"]["job_id"], settings)
