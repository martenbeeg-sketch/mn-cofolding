from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from .jobs import _read_json
from .runtime import Settings


def af2_msa_key(fasta_text: str, msa_mode: str) -> str:
    records: list[str] = []
    chunks: list[str] = []
    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if chunks:
                records.append("".join(chunks).upper())
            chunks = []
        else:
            chunks.append("".join(line.split()))
    if chunks:
        records.append("".join(chunks).upper())
    value = f"mn-cofolding-af2-msa-v1\0{msa_mode}\0" + "\0".join(records)
    return hashlib.sha256(value.encode()).hexdigest()


def af2_msa_path(settings: Settings, msa_mode: str, key: str) -> Path:
    return msa_cache_root(settings) / "mn-cofolding" / "af2" / msa_mode / f"{key}.a3m"


def msa_cache_root(settings: Settings) -> Path:
    return settings.msa_cache_dir or (settings.reference_dir / "msa_cache")


def normalized_protein_sequence(sequence: str) -> str:
    return "".join(str(sequence or "").split()).upper()


def shared_msa_path(sequence: str, settings: Settings) -> Path:
    cleaned = normalized_protein_sequence(sequence)
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return msa_cache_root(settings) / f"{digest}.a3m"


def a3m_query_sequence(text: str) -> str:
    in_query = False
    query: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if in_query:
                break
            in_query = True
        elif in_query:
            query.extend(char for char in line if char.isupper())
    return "".join(query).replace("-", "")


def read_matching_a3m(path: Path, sequence: str) -> str | None:
    try:
        content = path.read_text(encoding="utf-8").rstrip("\x00").replace("\r\n", "\n")
    except (OSError, UnicodeDecodeError):
        return None
    if "\x00" in content or not content.startswith(">"):
        return None
    return content if a3m_query_sequence(content) == normalized_protein_sequence(sequence) else None


def find_shared_msa(sequence: str, settings: Settings) -> tuple[Path | None, str]:
    cleaned = normalized_protein_sequence(sequence)
    if not cleaned:
        return None, "empty sequence"
    root = msa_cache_root(settings)
    canonical = shared_msa_path(cleaned, settings)
    if canonical.is_file() and read_matching_a3m(canonical, cleaned) is not None:
        return canonical, "sequence_hash"
    if root.is_dir():
        for candidate in sorted(root.glob("**/*.a3m")):
            if candidate == canonical:
                continue
            content = read_matching_a3m(candidate, cleaned)
            if content is not None:
                # Promote app-specific/legacy cache hits to the flat shared name
                # consumed directly by Protein Design and Ligand.
                return store_shared_msa(cleaned, content, settings), "sequence_scan"
    return None, "missing"


def store_shared_msa(sequence: str, content: str, settings: Settings) -> Path:
    cleaned = normalized_protein_sequence(sequence)
    if not cleaned or a3m_query_sequence(content) != cleaned:
        raise ValueError("A3M query sequence does not match the protein sequence")
    destination = shared_msa_path(cleaned, settings)
    if read_matching_a3m(destination, cleaned) is not None:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content.rstrip("\x00").replace("\r\n", "\n"), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def cache_af2_msa(run_dir: Path, settings: Settings) -> Path | None:
    request = _read_json(run_dir / "input.json")
    key = str(request.get("msa_cache_key") or "")
    mode = str(request.get("msa_mode") or "")
    if not key or mode in {"", "single_sequence"}:
        return None
    candidates = sorted((run_dir / "output").glob("*.a3m"))
    if len(candidates) != 1:
        return None
    destination = af2_msa_path(settings, mode, key)
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".a3m", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_bytes(candidates[0].read_bytes())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    request_fasta = (run_dir / str(request.get("input_file") or "")).read_text(encoding="utf-8")
    records: list[str] = []
    current: list[str] = []
    for line in request_fasta.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                records.extend("".join(current).split(":"))
            current = []
        else:
            current.append("".join(line.split()).upper())
    if current:
        records.extend("".join(current).split(":"))
    sequences = [normalized_protein_sequence(sequence) for sequence in records if sequence]
    if len(sequences) == 1:
        content = read_matching_a3m(candidates[0], sequences[0])
        if content is not None:
            store_shared_msa(sequences[0], content, settings)
    return destination
