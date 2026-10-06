from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .msa_cache import find_shared_msa, msa_cache_root, store_shared_msa
from .runtime import Settings


ALPHAFAST_IMAGE = "mn-alphafast:cu128"


def _request_proteins(run_dir: Path, request: dict) -> tuple[Path, list[tuple[str, str]]]:
    input_path = run_dir / str(request["input_file"])
    if request.get("input_format") == "fasta":
        sequence_parts: list[str] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(">"):
                sequence_parts.extend(part.upper() for part in line.split(":") if part)
        return input_path, [(sequence, "") for sequence in dict.fromkeys(sequence_parts)]

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    proteins: list[tuple[str, str]] = []
    for entity in payload.get("sequences") or []:
        protein = entity.get("protein") if isinstance(entity, dict) else None
        if not isinstance(protein, dict):
            continue
        sequence = "".join(str(protein.get("sequence") or "").split()).upper()
        if sequence:
            proteins.append((sequence, str(protein.get("unpairedMsa") or "")))
    return input_path, proteins


def _write_alphafast_input(path: Path, sequence: str, name: str) -> None:
    path.write_text(
        json.dumps(
            {
                "dialect": "alphafold3",
                "version": 4,
                "name": name,
                "sequences": [{"protein": {"id": "A", "sequence": sequence, "templates": []}}],
                "modelSeeds": [1],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _extract_unpaired_msas(output_dir: Path, sequences: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for data_path in sorted(output_dir.glob("**/*_data.json")):
        try:
            payload = json.loads(data_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for entity in payload.get("sequences") or []:
            protein = entity.get("protein") if isinstance(entity, dict) else None
            if not isinstance(protein, dict):
                continue
            sequence = "".join(str(protein.get("sequence") or "").split()).upper()
            msa = str(protein.get("unpairedMsa") or "").rstrip("\x00").replace("\r\n", "\n")
            if sequence in sequences and msa:
                found[sequence] = msa if msa.endswith("\n") else msa + "\n"
    return found


def _run_local_mmseqs(
    run_dir: Path,
    sequences: list[str],
    *,
    gpu_id: int,
    settings: Settings,
    runner=subprocess.run,
) -> dict[str, str]:
    db_dir = settings.reference_dir / "alignment"
    mmseqs_db = db_dir / "mmseqs"
    if not db_dir.is_dir() or not mmseqs_db.is_dir():
        raise RuntimeError(
            f"Local MMseqs-GPU needs the AlphaFast databases at {db_dir} and {mmseqs_db}"
        )

    artifact_dir = run_dir / "artifacts" / "local_mmseqs_msa"
    input_dir = artifact_dir / "input"
    output_dir = artifact_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, sequence in enumerate(sequences, start=1):
        digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
        _write_alphafast_input(input_dir / f"{digest}.json", sequence, f"msa_{index}_{digest}")

    command = [
        settings.docker_executable,
        "run",
        "--rm",
        "--gpus",
        f"device={gpu_id}",
        "--mount",
        f"type=bind,src={run_dir.resolve()},dst=/work",
        "--mount",
        f"type=bind,src={db_dir.resolve()},dst=/data/public_databases,readonly",
        "--mount",
        f"type=bind,src={mmseqs_db.resolve()},dst=/data/mmseqs_databases,readonly",
        "--env",
        "OMP_NUM_THREADS=4",
        "-w",
        "/app/alphafold",
        os.getenv("MN_COFOLDING_ALPHAFAST_IMAGE", ALPHAFAST_IMAGE),
        "python",
        "/app/alphafold/run_data_pipeline.py",
        "--input_dir=/work/artifacts/local_mmseqs_msa/input",
        "--output_dir=/work/artifacts/local_mmseqs_msa/output",
        "--db_dir=/data/public_databases",
        "--mmseqs_db_dir=/data/mmseqs_databases",
        "--use_mmseqs_gpu",
        "--batch_size=1",
    ]
    with (run_dir / "stdout.log").open("a", encoding="utf-8") as stdout, (run_dir / "stderr.log").open(
        "a", encoding="utf-8"
    ) as stderr:
        stdout.write(f"$ {' '.join(command)}\n")
        stdout.write(f"Generating {len(sequences)} missing MSA(s) with local MMseqs-GPU.\n")
        stdout.flush()
        completed = runner(command, stdout=stdout, stderr=stderr, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Local MMseqs-GPU MSA generation failed with return code {completed.returncode}")
    return _extract_unpaired_msas(output_dir, set(sequences))


def prepare_local_msa_inputs(
    run_dir: Path,
    request: dict,
    *,
    gpu_id: int | None,
    settings: Settings,
    runner=subprocess.run,
) -> dict:
    """Fill misses from shared AlphaFast/MMseqs-GPU cache for a queued run."""
    input_path, protein_entries = _request_proteins(run_dir, request)
    sequences = list(dict.fromkeys(sequence for sequence, _inline in protein_entries))
    if not sequences:
        raise ValueError("Local MMseqs-GPU needs at least one protein chain")
    if gpu_id is None:
        raise ValueError("Local MMseqs-GPU requires a reserved GPU")

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for sequence in sequences:
        cached, _source = find_shared_msa(sequence, settings)
        if cached is not None:
            resolved[sequence] = cached
        else:
            missing.append(sequence)
    if missing:
        generated = _run_local_mmseqs(run_dir, missing, gpu_id=gpu_id, settings=settings, runner=runner)
        for sequence in missing:
            msa_text = generated.get(sequence)
            if not msa_text:
                raise RuntimeError(f"Local MMseqs-GPU produced no MSA for sequence {sequence[:12]}…")
            resolved[sequence] = store_shared_msa(sequence, msa_text, settings)

    if request.get("input_format") == "fasta":
        if len(sequences) != 1:
            raise ValueError("Local MMseqs-GPU input for AlphaFold 2 currently supports one protein chain per job")
        relative = resolved[sequences[0]].resolve().relative_to(msa_cache_root(settings).resolve()).as_posix()
        return {"local_msa_path": relative, "engine_input_file": request["input_file"]}

    if request.get("input_format") == "alphafold3_json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        for entity in payload.get("sequences") or []:
            protein = entity.get("protein") if isinstance(entity, dict) else None
            if not isinstance(protein, dict):
                continue
            sequence = "".join(str(protein.get("sequence") or "").split()).upper()
            if sequence not in resolved:
                continue
            protein["unpairedMsa"] = resolved[sequence].read_text(encoding="utf-8")
            protein["pairedMsa"] = ""
            protein.setdefault("templates", [])
        engine_input = run_dir / "input" / "engine-input.json"
        engine_input.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {
            "engine_input_file": engine_input.relative_to(run_dir).as_posix(),
            "local_msa_path": "",
        }

    raise ValueError("Local MMseqs-GPU is supported for protein FASTA or AlphaFold 3 JSON inputs")
