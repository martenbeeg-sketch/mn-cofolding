from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .models import get_model


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")
NUCLEOTIDES_DNA = frozenset("ACGTNRYKMSWBDHV")
NUCLEOTIDES_RNA = frozenset("ACGUNRYKMSWBDHV")
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_SEQUENCE_RESIDUES = 100_000


@dataclass(frozen=True)
class CofoldInput:
    text: str
    file_name: str
    input_format: str
    chain_count: int
    residue_count: int


def _split_entries(value: str) -> list[str]:
    return [entry for entry in re.split(r":+", value.strip().replace("\n", "").replace("\r", "")) if entry]


def _check_sequence(value: str, alphabet: frozenset[str], molecule: str) -> str:
    sequence = "".join(value.split()).upper()
    if not sequence:
        raise ValueError(f"{molecule} sequences cannot be empty")
    invalid = sorted(set(sequence) - alphabet)
    if invalid:
        raise ValueError(f"Invalid character(s) in {molecule} sequence: {''.join(invalid)}")
    if len(sequence) > MAX_SEQUENCE_RESIDUES:
        raise ValueError(f"A {molecule} sequence exceeds {MAX_SEQUENCE_RESIDUES} residues")
    return sequence


def build_af3_json(
    *,
    name: str,
    protein: str = "",
    dna: str = "",
    rna: str = "",
    ligand_ccd: str = "",
    ligand_smiles: str = "",
    seeds: list[int] | None = None,
    msa_mode: str = "mmseqs2_uniref_env",
) -> str:
    """Create the AlphaFold 3 JSON input used by the preview notebook."""
    sequences: list[dict] = []
    chain_ids = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    residue_count = 0

    def add(kind: str, sequence: str, *, field: str = "sequence") -> None:
        nonlocal residue_count
        try:
            chain_id = next(chain_ids)
        except StopIteration as exc:
            raise ValueError("An input cannot contain more than 52 chains or ligands") from exc
        content: dict[str, object] = {"id": [chain_id], field: [sequence] if field == "ccdCodes" else sequence}
        if kind == "protein":
            content["templates"] = []
        if kind in {"protein", "rna"} and msa_mode == "single_sequence":
            content["unpairedMsa"] = f">query\n{sequence}\n"
            if kind == "protein":
                content["pairedMsa"] = ""
        sequences.append({kind: content})
        residue_count += len(sequence) if field == "sequence" else 0

    for entry in _split_entries(protein):
        add("protein", _check_sequence(entry, AMINO_ACIDS, "Protein"))
    for entry in _split_entries(rna):
        add("rna", _check_sequence(entry, NUCLEOTIDES_RNA, "RNA"))
    for entry in _split_entries(dna):
        add("dna", _check_sequence(entry, NUCLEOTIDES_DNA, "DNA"))
    for code in _split_entries(ligand_ccd):
        if not re.fullmatch(r"[A-Za-z0-9]{1,8}", code):
            raise ValueError(f"Invalid CCD code {code!r}; use one or more 1–8 character component IDs")
        add("ligand", code.upper(), field="ccdCodes")
    for smiles in _split_entries(ligand_smiles):
        if len(smiles) > 50_000:
            raise ValueError("A ligand SMILES string exceeds 50000 characters")
        add("ligand", smiles, field="smiles")
    if not sequences:
        raise ValueError("Add at least one protein, DNA, RNA, CCD ligand, or SMILES ligand")
    seed_values = list(dict.fromkeys(int(seed) for seed in (seeds or [1])))
    if not seed_values or any(seed < 0 for seed in seed_values):
        raise ValueError("Provide one or more non-negative random seeds")
    payload = {
        "name": re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:80] or "cofold",
        "sequences": sequences,
        "modelSeeds": seed_values,
        "dialect": "alphafold3",
        "version": 1,
    }
    return json.dumps(payload, indent=2) + "\n"


def validate_input(text: str, input_format: str, model_name: str) -> CofoldInput:
    get_model(model_name)
    text = text.strip()
    if not text:
        raise ValueError("Input cannot be empty")
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError(f"Input exceeds the {MAX_INPUT_BYTES // 1024 // 1024} MiB limit")
    if input_format == "fasta":
        records: list[tuple[str, str]] = []
        header = "query"
        chunks: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if chunks:
                    records.append((header, "".join(chunks)))
                header = line[1:].strip() or f"query_{len(records) + 1}"
                chunks = []
            else:
                if not chunks and not records and header == "query" and not text.lstrip().startswith(">"):
                    # Accept a bare sequence in the UI as well as FASTA.
                    header = "query"
                elif not chunks and text.lstrip().startswith(">") and not line.startswith(">"):
                    pass
                chunks.append(line)
        if chunks:
            records.append((header, "".join(chunks)))
        if not records:
            raise ValueError("FASTA input needs at least one sequence")
        chains = 0
        residues = 0
        for _record_name, sequence in records:
            for chain in sequence.split(":"):
                cleaned = _check_sequence(chain, AMINO_ACIDS, "Protein")
                chains += 1
                residues += len(cleaned)
        normalized = "".join(f">{record_name}\n{sequence.upper()}\n" for record_name, sequence in records)
        return CofoldInput(normalized, "input.fasta", "fasta", chains, residues)
    if input_format == "alphafold3_json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Input is not valid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("sequences"), list):
            raise ValueError("AlphaFold 3 JSON must be an object containing a sequences list")
        if not payload["sequences"]:
            raise ValueError("AlphaFold 3 JSON contains no molecular sequences")
        return CofoldInput(json.dumps(payload, indent=2) + "\n", "input.json", "alphafold3_json", len(payload["sequences"]), 0)
    raise ValueError("input_format must be 'fasta' or 'alphafold3_json'")


def validate_model_input_compatibility(model_name: str, input_format: str) -> None:
    model = get_model(model_name)
    if model.family == "af2" and input_format != "fasta":
        raise ValueError("AlphaFold 2 models accept protein FASTA input only")


def fasta_to_af3_json(
    value: CofoldInput,
    *,
    name: str,
    seeds: list[int],
    msa_mode: str = "mmseqs2_uniref_env",
) -> CofoldInput:
    """Convert one protein FASTA record to the AF3 JSON dialect required by run_alphafold.py."""
    records: list[tuple[str, str]] = []
    header = "query"
    parts: list[str] = []
    for line in value.text.splitlines():
        if line.startswith(">"):
            if parts:
                records.append((header, "".join(parts)))
            header, parts = line[1:] or "query", []
        elif line.strip():
            parts.append(line.strip())
    if parts:
        records.append((header, "".join(parts)))
    if len(records) != 1:
        raise ValueError("Submit one complex per job. Use colon-separated chains in one FASTA record.")
    header, sequence = records[0]
    chains = []
    for index, chain in enumerate(sequence.split(":")):
        clean = _check_sequence(chain, AMINO_ACIDS, "Protein")
        chain_id = chr(ord("A") + index) if index < 26 else f"A{index - 25}"
        protein = {"id": [chain_id], "sequence": clean, "modifications": [], "templates": []}
        if msa_mode == "single_sequence":
            protein.update({"unpairedMsa": f">query\n{clean}\n", "pairedMsa": ""})
        chains.append({"protein": protein})
    payload = {
        "name": re.sub(r"[^A-Za-z0-9_-]+", "_", name or header).strip("_")[:80] or "cofold",
        "sequences": chains,
        "modelSeeds": list(dict.fromkeys(seeds or [1])),
        "dialect": "alphafold3",
        "version": 1,
    }
    return CofoldInput(json.dumps(payload, indent=2) + "\n", "input.json", "alphafold3_json", len(chains), sum(len(item["protein"]["sequence"]) for item in chains))
