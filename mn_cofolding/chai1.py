from __future__ import annotations

import json
import re
from pathlib import Path


def _ids(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    elif value is None:
        raw = []
    else:
        raw = [str(value)]
    return [item for item in raw if item]


def prepare_chai1_input(run_dir: Path, request: dict) -> dict:
    """Convert an AF3-shaped protein input and inline A3Ms to Chai's FASTA/items format."""
    source = request.get("engine_input_file") or request["input_file"]
    payload = json.loads((run_dir / str(source)).read_text(encoding="utf-8"))
    use_msa = str(request.get("msa_source")) == "local_mmseqs_gpu" and request.get("msa_mode") != "single_sequence"
    msa_dir = run_dir / "input" / "chai1_msas"
    if msa_dir.exists():
        for stale in msa_dir.glob("*"):
            if stale.is_file():
                stale.unlink()
    msa_dir.mkdir(parents=True, exist_ok=True)
    records: list[str] = []
    chain_count = 0
    msa_depth = int(request.get("msa_max_depth", 1024))

    for entity in payload.get("sequences") or []:
        protein = entity.get("protein") if isinstance(entity, dict) else None
        if not isinstance(protein, dict):
            raise ValueError("Native Chai-1 accepts protein chains only")
        raw_sequence = protein.get("sequence") or ""
        sequence = "".join(map(str, raw_sequence)) if isinstance(raw_sequence, list) else str(raw_sequence)
        sequence = "".join(sequence.split()).upper()
        if not sequence:
            raise ValueError("Chai-1 input contains an empty protein sequence")
        ids = _ids(protein.get("id")) or [chr(ord("A") + chain_count)]
        msa_text = str(protein.get("unpairedMsa") or "") if use_msa else ""
        if use_msa and not msa_text.strip():
            raise ValueError(f"Chai-1 chain {ids[0]} has no prepared local MSA")
        for chain_id in ids:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", chain_id) or f"chain_{chain_count}"
            records.extend((f">protein|name={name}", sequence))
            if msa_text:
                (msa_dir / f"{name}.a3m").write_text(msa_text.rstrip() + "\n", encoding="utf-8")
            chain_count += 1

    if not records:
        raise ValueError("Chai-1 needs at least one protein chain")
    fasta_path = run_dir / "input" / "chai1.fasta"
    fasta_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    items_path = run_dir / "input" / "chai1-input.json"
    item = {
        "id": str(payload.get("name") or request.get("name") or request.get("job_id") or "cofold"),
        "fasta": fasta_path.name,
        "seeds": list(request.get("seeds") or [1]),
    }
    items = {"items": [item]}
    if use_msa:
        items["msa_dir"] = msa_dir.name
    items_path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    return {
        "chai1_input_file": items_path.relative_to(run_dir).as_posix(),
        "chai1_fasta_file": fasta_path.relative_to(run_dir).as_posix(),
        "chai1_msa_dir": msa_dir.relative_to(run_dir).as_posix() if use_msa else None,
        "chai1_chain_count": chain_count,
        "chai1_msa_depth": msa_depth,
    }
