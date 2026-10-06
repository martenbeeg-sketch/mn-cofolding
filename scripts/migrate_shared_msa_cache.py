#!/usr/bin/env python3
"""Move known MSA repositories into one reference_files/msa_cache directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _move_tree(source: Path, destination: Path, source_label: str) -> int:
    moved = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        if target.is_file():
            if _sha256(path) == _sha256(target):
                path.unlink()
                continue
            target = destination / "sources" / source_label / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_file() and _sha256(path) == _sha256(target):
                path.unlink()
                continue
            suffix = _sha256(path)[:12]
            target = target.with_name(f"{target.stem}-{suffix}{target.suffix}")
        os.replace(path, target)
        moved += 1
    shutil.rmtree(source)
    source.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(destination, source.parent)
    source.symlink_to(relative_target, target_is_directory=True)
    return moved


def _normalize_sequence(value: object) -> str:
    return "".join(str(value or "").split()).upper()


def _a3m_query_sequence(text: str) -> str:
    seen_header = False
    sequence: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seen_header:
                break
            seen_header = True
        elif seen_header:
            sequence.extend(char for char in line if char.isupper())
    return "".join(sequence).replace("-", "")


def _import_benchmark_prefill(reference_root: Path, msa_root: Path) -> int:
    prefill = reference_root / "de_novo_binder_scoring_overath_2025" / "target_msa_prefill"
    imported = 0
    for path in sorted(prefill.glob("**/*_data.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for entity in payload.get("sequences") or []:
            protein = entity.get("protein") if isinstance(entity, dict) else None
            if not isinstance(protein, dict):
                continue
            sequence = _normalize_sequence(protein.get("sequence"))
            content = str(protein.get("unpairedMsa") or "").rstrip("\x00").replace("\r\n", "\n")
            if not sequence or _a3m_query_sequence(content) != sequence:
                continue
            if len([line for line in content.splitlines() if line.startswith(">")]) < 2:
                continue
            destination = msa_root / f"{hashlib.sha256(sequence.encode('utf-8')).hexdigest()}.a3m"
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
            imported += 1
    return imported


def migrate(reference_root: Path, *, include_benchmark_prefill: bool = True) -> dict[str, int]:
    reference_root = reference_root.expanduser().resolve()
    msa_root = reference_root / "msa_cache"
    msa_root.mkdir(parents=True, exist_ok=True)
    migrated = 0

    migrations = (
        (reference_root / "boltz_models" / "msa_repository", msa_root, "boltz_models"),
        (reference_root / "mn-cofolding" / "cache" / "msa", msa_root / "mn-cofolding", "mn-cofolding"),
    )
    for source, destination, label in migrations:
        if source.is_symlink() or not source.is_dir():
            continue
        migrated += _move_tree(source, destination, label)

    prefill_count = _import_benchmark_prefill(reference_root, msa_root) if include_benchmark_prefill else 0
    return {"files_moved": migrated, "benchmark_msas_imported": prefill_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path("/mnt/db/reference_files"))
    parser.add_argument("--skip-benchmark-prefill", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Move old cache folders and add backward-compatible symlinks")
    args = parser.parse_args()
    if not args.apply:
        print("Dry run only: rerun with --apply to migrate cache contents.")
        print(f"Reference root: {args.reference_root.expanduser().resolve()}")
        return 0
    result = migrate(args.reference_root, include_benchmark_prefill=not args.skip_benchmark_prefill)
    print(json.dumps(result, indent=2))
    print(f"Shared cache: {args.reference_root.expanduser().resolve() / 'msa_cache'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
