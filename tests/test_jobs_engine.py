from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mn_cofolding.engine import docker_command, gpu_tuning
from mn_cofolding.esmfold2 import prepare_esmfold2_input
from mn_cofolding.jobs import JobStore
from mn_cofolding.msa_cache import af2_msa_path
from mn_cofolding.models import MODELS, image_for_model


EXAMPLE_FASTA = ">test\nPIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK\n"


def test_job_store_converts_fasta_to_runner_json_and_stores_all_inputs(settings):
    store = JobStore(settings)
    job = store.create(model_name="openfold3", input_text=EXAMPLE_FASTA, seeds=[7, 4])
    detail = store.get(job["metadata"]["job_id"])
    run_dir = detail["run_dir"]
    assert detail["metadata"]["status"] == "queued"
    assert detail["input"]["seeds"] == [7, 4]
    assert json.loads((run_dir / detail["input"]["input_file"]).read_text())["modelSeeds"] == [7, 4]
    assert (run_dir / "stdout.log").is_file()
    assert (run_dir / "output").is_dir()


def test_job_defaults_match_alphafold3_run_settings(settings):
    job = JobStore(settings).create(model_name="openfold3", input_text=EXAMPLE_FASTA)
    assert job["input"]["num_recycles"] == 10
    assert job["input"]["num_diffusion_samples"] == 5
    assert job["input"]["optimization_mode"] == "off"


def test_app_default_af3_image_is_the_shared_mode_image(monkeypatch, tmp_path):
    from mn_cofolding.runtime import Settings

    monkeypatch.delenv("MN_COFOLDING_AF3_IMAGE", raising=False)
    monkeypatch.setenv("MN_COFOLDING_APP_HOME", str(tmp_path / "app"))
    assert Settings.from_env().af3_image == "mn-cofolding-af3-memory:3.1.14-cu13"


@pytest.mark.parametrize("mode", ["off", "fast", "big"])
def test_af3_optimization_modes_are_saved_and_passed_to_the_runner(settings, monkeypatch, mode):
    store = JobStore(settings)
    job = store.create(model_name="openfold3", input_text=EXAMPLE_FASTA, optimization_mode=mode)
    assert job["input"]["optimization_mode"] == mode
    monkeypatch.setattr("mn_cofolding.engine.gpu_tuning", lambda _gpu_id: ("xla", ""))
    command = docker_command(job["run_dir"], gpu_id=0, settings=settings)
    mode_env = f"MN_COFOLDING_AF3_OPT_MODE={mode}"
    assert command[command.index(mode_env) - 1] == "--env"


def test_optimization_modes_are_limited_to_af3_family(settings):
    store = JobStore(settings)
    with pytest.raises(ValueError, match="AF3-family"):
        store.create(model_name="af2_ptm", input_text=EXAMPLE_FASTA, optimization_mode="fast")
    with pytest.raises(ValueError, match="optimization_mode"):
        store.create(model_name="openfold3", input_text=EXAMPLE_FASTA, optimization_mode="turbo")
    with pytest.raises(ValueError, match="reserved GPU"):
        store.create(model_name="openfold3", input_text=EXAMPLE_FASTA, optimization_mode="big", gpu=False)


@pytest.mark.parametrize("mode", ["fast", "big"])
def test_stock_af3_image_rejects_fast_and_big(settings, monkeypatch, mode):
    from dataclasses import replace

    stock_settings = replace(settings, af3_image="mn-cofolding-af3:3.1.14-cu13")
    job = JobStore(stock_settings).create(
        model_name="openfold3", input_text=EXAMPLE_FASTA, optimization_mode=mode
    )
    monkeypatch.setattr("mn_cofolding.engine.gpu_tuning", lambda _gpu_id: ("xla", ""))
    with pytest.raises(ValueError, match="stock image supports off only"):
        docker_command(job["run_dir"], gpu_id=0, settings=stock_settings)


def test_official_af3_requires_explicit_terms_acknowledgement(settings):
    with pytest.raises(ValueError, match="terms"):
        JobStore(settings).create(model_name="alphafold3", input_text=EXAMPLE_FASTA)


def test_engine_routes_all_preview_models_to_one_image(settings, monkeypatch):
    store = JobStore(settings)
    for model in MODELS:
        expected_image = (
            settings.esmfold2_image
            if model.name == "esmfold2"
            else settings.af2_image
            if model.family == "af2"
            else settings.af3_image
        )
        assert image_for_model(model.name, af3_image=settings.af3_image, af2_image=settings.af2_image) == expected_image
    af3 = store.create(model_name="openfold3", input_text=EXAMPLE_FASTA)
    af2 = store.create(model_name="af2_ptm", input_text=EXAMPLE_FASTA, seeds=[7, 8], num_recycles=2)
    monkeypatch.setattr("mn_cofolding.engine.gpu_tuning", lambda _gpu_id: ("triton", "--xla_gpu_enable_triton_gemm=false"))
    af3_command = docker_command(af3["run_dir"], gpu_id=1, settings=settings)
    af2_command = docker_command(af2["run_dir"], gpu_id=1, settings=settings)
    assert settings.af3_image in af3_command
    assert "/usr/local/bin/cofolding-af3-runner.py" in af3_command
    assert "--model=openfold3" in af3_command
    assert "--force_output_dir" in af3_command
    assert "--cache_dir=/cache/alphafold3" in af3_command
    assert "AF3_WEIGHTS_DIR=/cache/alphafold3/weights" in af3_command
    assert f"type=bind,src={settings.msa_cache_dir.resolve()},dst=/msa_cache" in af3_command
    assert "AF3_SHARED_MSA_CACHE_DIR=/msa_cache" in af3_command
    assert "--num_recycles=10" in af3_command
    assert "--num_diffusion_samples=5" in af3_command
    assert "--cpus" not in af3_command
    assert "OMP_NUM_THREADS=4" in af3_command
    assert str(settings.cache_dir.resolve()) in " ".join(af3_command)
    assert settings.af2_image in af2_command
    assert "--model-type" in af2_command
    assert af2_command[af2_command.index("--model-type") + 1] == "alphafold2_ptm"
    assert af2_command[af2_command.index("--random-seed") + 1] == "7"
    assert af2_command[af2_command.index("--num-seeds") + 1] == "2"
    assert af2_command[af2_command.index("--num-recycle") + 1] == "2"
    assert str(settings.cache_dir.resolve()) in " ".join(af2_command)
    assert str((settings.reference_dir / "alphafold_models").resolve()) in " ".join(af2_command)


def test_af2_requires_consecutive_seed_values(settings):
    with pytest.raises(ValueError, match="consecutive"):
        JobStore(settings).create(model_name="af2_multimer", input_text=EXAMPLE_FASTA, seeds=[1, 3])


def test_af2_reuses_cached_msa_as_input(settings):
    store = JobStore(settings)
    job = store.create(model_name="af2_ptm", input_text=EXAMPLE_FASTA)
    request = job["input"]
    assert request["msa_cache_key"]
    cache_file = af2_msa_path(settings, request["msa_mode"], request["msa_cache_key"])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(">query\nPIAQI\n")

    command = docker_command(job["run_dir"], gpu_id=0, settings=settings)
    assert f"/msa_cache/mn-cofolding/af2/{request['msa_mode']}/{request['msa_cache_key']}.a3m" in command
    assert f"type=bind,src={settings.msa_cache_dir.resolve()},dst=/msa_cache" in command
    assert "AF3_SHARED_MSA_CACHE_DIR=/msa_cache" not in command


@pytest.mark.parametrize(("cap", "expected"), [("7.5", "xla"), ("8.9", "xla"), ("9.0", "triton"), ("12.0", "triton")])
def test_gpu_tuning_matches_preview_compute_capability_rules(cap, expected):
    runner = lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=f"{cap}\n")
    attention, _flags = gpu_tuning(0, runner=runner)
    assert attention == expected


def test_no_gpu_defaults_to_xla():
    assert gpu_tuning(None) == ("xla", "")


def test_esmfold2_full_weights_modes_prepare_input_and_route_gpu(settings, monkeypatch):
    from mn_cofolding.inputs import build_af3_json
    from mn_cofolding.jobs import _write_json

    text = build_af3_json(
        name="test_dimer",
        protein="PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK:PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        msa_mode="single_sequence",
    )
    store = JobStore(settings)
    job = store.create(
        model_name="esmfold2",
        input_text=text,
        input_format="alphafold3_json",
        msa_mode="single_sequence",
        msa_source="single_sequence",
        optimization_mode="exact",
        seeds=[0, 1],
        num_diffusion_samples=2,
        msa_max_depth=1024,
    )
    request = job["input"]
    request.update(prepare_esmfold2_input(job["run_dir"], request))
    _write_json(job["run_dir"] / "input.json", request)
    monkeypatch.setattr("mn_cofolding.engine.esmfold2_config", lambda _gpu_id: "rtx4090")

    command = docker_command(job["run_dir"], gpu_id=0, settings=settings)

    assert settings.esmfold2_image in command
    assert "mn-cofolding-af3-runner.py" not in command
    assert "--config" in command and command[command.index("--config") + 1] == "rtx4090"
    assert command[command.index("--variant") + 1] == "full_nomsa"
    assert command[command.index("--mode") + 1] == "exact"
    assert command[command.index("--n_gpu") + 1] == "1"
    assert command[command.index("--seeds") + 1] == "0,1"
    assert command[command.index("--msa_max_depth") + 1] == "1024"
    converted = json.loads((job["run_dir"] / request["esmfold2_input_file"]).read_text())
    assert len(converted["sequences"]) == 2
    assert all("msa" not in chain for chain in converted["sequences"])


def test_esmfold2_big_reserves_two_gpus_only_for_big(settings, monkeypatch):
    from mn_cofolding.inputs import build_af3_json
    from mn_cofolding.jobs import _write_json

    input_text = build_af3_json(name="test", protein="PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK", msa_mode="single_sequence")
    job = JobStore(settings).create(
        model_name="esmfold2",
        input_text=input_text,
        input_format="alphafold3_json",
        msa_mode="single_sequence",
        msa_source="single_sequence",
        optimization_mode="big",
    )
    request = job["input"]
    request.update(prepare_esmfold2_input(job["run_dir"], request))
    _write_json(job["run_dir"] / "input.json", request)
    monkeypatch.setattr("mn_cofolding.engine.esmfold2_config", lambda _gpu_id: "rtx5090")

    command = docker_command(job["run_dir"], gpu_ids=[0, 1], settings=settings)

    assert job["input"]["gpu_count"] == 2
    assert command[command.index("--config") + 1] == "rtx5090"
    assert command[command.index("--gpus") + 1] == '"device=0,1"'
    assert command[command.index("--n_gpu") + 1] == "2"
    assert command[command.index("--mode") + 1] == "big"
