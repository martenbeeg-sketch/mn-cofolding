from __future__ import annotations

import json
import re
from pathlib import Path


def _chain_ids(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    elif value is None:
        values = []
    else:
        values = [str(value)]
    return [item for item in values if item]


def prepare_esmfold2_input(run_dir: Path, request: dict) -> dict:
    """Convert the app's AF3-shaped protein/MSA input to the kit's native input schema."""
    source = request.get("engine_input_file") or request["input_file"]
    source_path = run_dir / str(source)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    chains: list[dict] = []
    msa_dir = run_dir / "input" / "esmfold2_msas"
    msa_dir.mkdir(parents=True, exist_ok=True)
    use_msa = str(request.get("msa_source")) == "local_mmseqs_gpu" and request.get("msa_mode") != "single_sequence"

    for entity in payload.get("sequences") or []:
        protein = entity.get("protein") if isinstance(entity, dict) else None
        if not isinstance(protein, dict):
            raise ValueError("ESMFold2 accepts protein chains only")
        sequence = "".join(str(protein.get("sequence") or "").split()).upper()
        if not sequence:
            raise ValueError("ESMFold2 input contains an empty protein sequence")
        ids = _chain_ids(protein.get("id")) or [chr(ord("A") + len(chains))]
        msa_text = str(protein.get("unpairedMsa") or "") if use_msa else ""
        if use_msa and not msa_text.strip():
            raise ValueError(f"ESMFold2 chain {ids[0]} has no prepared local MSA")
        for chain_id in ids:
            chain = {"type": "protein", "id": chain_id, "sequence": sequence}
            if msa_text:
                msa_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", chain_id) or f"chain_{len(chains)}"
                msa_path = msa_dir / f"{msa_name}.a3m"
                msa_path.write_text(msa_text.rstrip() + "\n", encoding="utf-8")
                chain["msa"] = msa_path.relative_to(source_path.parent).as_posix()
            chains.append(chain)

    if not chains:
        raise ValueError("ESMFold2 needs at least one protein chain")
    kit_input = {
        "id": str(payload.get("name") or request.get("name") or request.get("job_id") or "cofold"),
        "sequences": chains,
        "seeds": list(request.get("seeds") or [1]),
        "num_diffusion_samples": int(request.get("num_diffusion_samples", 5)),
    }
    destination = run_dir / "input" / "esmfold2-input.json"
    destination.write_text(json.dumps(kit_input, indent=2) + "\n", encoding="utf-8")
    return {
        "esmfold2_input_file": destination.relative_to(run_dir).as_posix(),
        "esmfold2_variant": "full_msa" if use_msa else "full_nomsa",
        "esmfold2_chain_count": len(chains),
    }
