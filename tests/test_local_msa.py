from __future__ import annotations

import json
import hashlib
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import pytest

from mn_cofolding.jobs import JobStore
from mn_cofolding.local_msa import prepare_local_msa_inputs
from mn_cofolding.msa_cache import find_shared_msa, shared_msa_path
from mn_cofolding.engine import docker_command

_MIGRATION_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_shared_msa_cache.py"
_MIGRATION_SPEC = importlib.util.spec_from_file_location("msa_cache_migration", _MIGRATION_SCRIPT)
_MIGRATION_MODULE = importlib.util.module_from_spec(_MIGRATION_SPEC)
assert _MIGRATION_SPEC and _MIGRATION_SPEC.loader
_MIGRATION_SPEC.loader.exec_module(_MIGRATION_MODULE)
migrate = _MIGRATION_MODULE.migrate


EXAMPLE = ">local\nPIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK\n"


def test_local_mmseqs_generates_and_reuses_shared_msa(settings):
    db_dir = settings.reference_dir / "alignment"
    (db_dir / "mmseqs").mkdir(parents=True)
    job = JobStore(settings).create(
        model_name="openfold3",
        input_text=EXAMPLE,
        msa_source="local_mmseqs_gpu",
    )
    request = job["input"]
    calls = []

    def fake_runner(command, **_kwargs):
        calls.append(command)
        input_dir = job["run_dir"] / "artifacts" / "local_mmseqs_msa" / "input"
        output_dir = job["run_dir"] / "artifacts" / "local_mmseqs_msa" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        for input_path in input_dir.glob("*.json"):
            payload = json.loads(input_path.read_text())
            protein = payload["sequences"][0]["protein"]
            sequence = protein["sequence"]
            target = output_dir / input_path.stem
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{input_path.stem}_data.json").write_text(
                json.dumps({"sequences": [{"protein": {"sequence": sequence, "unpairedMsa": f">query\n{sequence}\n>local\n{sequence}\n"}}]})
            )
        return SimpleNamespace(returncode=0)

    prepared = prepare_local_msa_inputs(
        job["run_dir"], request, gpu_id=1, settings=settings, runner=fake_runner
    )
    engine_input = json.loads((job["run_dir"] / prepared["engine_input_file"]).read_text())
    msa = engine_input["sequences"][0]["protein"]["unpairedMsa"]
    sequence = engine_input["sequences"][0]["protein"]["sequence"]
    cached, source = find_shared_msa(sequence, settings)

    assert len(calls) == 1
    assert "--use_mmseqs_gpu" in calls[0]
    assert source == "sequence_hash"
    assert cached == shared_msa_path(sequence, settings)
    assert msa.startswith(f">query\n{sequence}\n")
    assert engine_input["sequences"][0]["protein"]["pairedMsa"] == ""

    prepare_local_msa_inputs(job["run_dir"], request, gpu_id=1, settings=settings, runner=fake_runner)
    assert len(calls) == 1


def test_nested_msa_hit_is_promoted_to_flat_shared_cache(settings):
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    nested = settings.msa_cache_dir / "mn-cofolding" / "alphafold3" / "unpaired" / "prior.a3m"
    nested.parent.mkdir(parents=True)
    nested.write_text(f">query\n{sequence}\n>hit\n{sequence}\n")

    cached, source = find_shared_msa(sequence, settings)

    assert source == "sequence_scan"
    assert cached == shared_msa_path(sequence, settings)
    assert cached.is_file()
    assert cached.read_text() == nested.read_text()


def test_local_mmseqs_job_requires_gpu_and_af2_single_chain(settings):
    with pytest.raises(ValueError, match="reserved GPU"):
        JobStore(settings).create(model_name="openfold3", input_text=EXAMPLE, msa_source="local_mmseqs_gpu", gpu=False)
    with pytest.raises(ValueError, match="one protein chain"):
        JobStore(settings).create(
            model_name="af2_multimer",
            input_text=">complex\nAAAA:BBBB\n",
            msa_source="local_mmseqs_gpu",
        )


def test_af2_engine_mounts_the_shared_a3m_path(settings):
    job = JobStore(settings).create(
        model_name="af2_ptm",
        input_text=EXAMPLE,
        msa_source="local_mmseqs_gpu",
    )
    request = job["input"]
    request["local_msa_path"] = "a" * 64 + ".a3m"
    (job["run_dir"] / "input.json").write_text(json.dumps(request))

    command = docker_command(job["run_dir"], gpu_id=0, settings=settings)

    assert "/msa_cache/" + request["local_msa_path"] in command


def test_migration_collects_legacy_and_benchmark_msas(tmp_path):
    repository_root = tmp_path / "references"
    old_protein_msa = repository_root / "boltz_models" / "msa_repository"
    old_protein_msa.mkdir(parents=True)
    (old_protein_msa / "legacy.a3m").write_text(">query\nAAAA\n>hit\nAAAA\n")
    old_cofold_msa = repository_root / "mn-cofolding" / "cache" / "msa" / "af2"
    old_cofold_msa.mkdir(parents=True)
    (old_cofold_msa / "saved.a3m").write_text(">query\nBBBB\n>hit\nBBBB\n")
    prefill = repository_root / "de_novo_binder_scoring_overath_2025" / "target_msa_prefill" / "alphafast_output"
    prefill.mkdir(parents=True)
    sequence = "CCCC"
    (prefill / "target_data.json").write_text(
        json.dumps({"sequences": [{"protein": {"sequence": sequence, "unpairedMsa": ">query\nCCCC\n>hit\nCCCC\n"}}]})
    )

    result = migrate(repository_root)
    msa_root = repository_root / "msa_cache"

    assert result == {"files_moved": 2, "benchmark_msas_imported": 1}
    assert old_protein_msa.is_symlink()
    assert old_protein_msa.resolve() == msa_root.resolve()
    assert old_cofold_msa.parent.is_symlink()
    assert (msa_root / "legacy.a3m").is_file()
    assert (msa_root / "mn-cofolding" / "af2" / "saved.a3m").is_file()
    assert (msa_root / f"{hashlib.sha256(sequence.encode()).hexdigest()}.a3m").is_file()
