from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .jobs import ACTIVE_STATUSES, JobStore, _read_json
from .runtime import Settings, get_settings


MANIFEST_NAME = "portable-export.json"
SCHEMA_VERSION = 1


class PortabilityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or path.parts[0] != "runs":
        raise PortabilityError(f"Unsafe path in archive: {value}")
    return path


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for dirname in list(dirnames):
            if (current / dirname).is_symlink():
                raise PortabilityError(f"Cannot export symbolic link: {current / dirname}")
        dirnames[:] = [name for name in dirnames if name != "__pycache__"]
        for filename in filenames:
            path = current / filename
            if path.is_symlink():
                raise PortabilityError(f"Cannot export symbolic link: {path}")
            if filename in {".worker-claim.json"}:
                continue
            if path.is_file():
                files.append(path)
    return sorted(files)


def export_workdir(output: str | Path, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    store = JobStore(settings)
    active = [item for item in store.list(limit=100_000) if item.get("status") in ACTIVE_STATUSES]
    if active:
        ids = ", ".join(str(item.get("job_id")) for item in active[:8])
        raise PortabilityError(f"Stop or finish active jobs before export: {ids}")
    source = settings.runs_dir / "cofolding"
    files = _iter_files(source)
    entries = []
    for path in files:
        rel = path.relative_to(settings.runs_dir).as_posix()
        entries.append({"path": rel, "size": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "application": "mn-cofolding",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_count": len({PurePosixPath(item["path"]).parts[1] for item in entries}),
        "files": entries,
        "references_included": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path("runs") / path.relative_to(settings.runs_dir))
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def export_job_archive(job_id: str, settings: Settings | None = None) -> bytes:
    """Build a self-contained ZIP for one finished job, including logs and inputs."""
    settings = settings or get_settings()
    store = JobStore(settings)
    detail = store.get(job_id)
    status = str(detail["metadata"].get("status") or "")
    if status in ACTIVE_STATUSES:
        raise PortabilityError("Wait for the job to finish before downloading its ZIP archive")
    run_dir = detail["run_dir"]
    files = _iter_files(run_dir)
    entries = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "application": "mn-cofolding-job",
        "job_id": job_id,
        "created_at": detail["metadata"].get("created_at"),
        "status": status,
        "files": entries,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path(job_id) / path.relative_to(run_dir))
        archive.writestr(
            (Path(job_id) / "job-export.json").as_posix(),
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    return buffer.getvalue()


def verify_archive(archive_path: str | Path) -> dict:
    path = Path(archive_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise PortabilityError(f"Corrupt archive member: {bad}")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise PortabilityError("Archive contains duplicate member names")
            for info in infos:
                if info.filename == MANIFEST_NAME:
                    continue
                _safe_archive_path(info.filename)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise PortabilityError(f"Archive contains a symbolic link: {info.filename}")
            try:
                manifest = json.loads(archive.read(MANIFEST_NAME))
            except (KeyError, json.JSONDecodeError) as exc:
                raise PortabilityError("Archive has no valid portability manifest") from exc
            if manifest.get("application") != "mn-cofolding" or manifest.get("schema_version") != SCHEMA_VERSION:
                raise PortabilityError("Unsupported portability archive format")
            expected = {str(item["path"]): item for item in manifest.get("files", [])}
            actual = {info.filename.removeprefix("runs/"): info for info in infos if info.filename != MANIFEST_NAME}
            if set(expected) != set(actual):
                raise PortabilityError("Archive contents do not match its manifest")
            for relative, entry in expected.items():
                data = archive.read(f"runs/{relative}")
                if len(data) != int(entry["size"]) or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise PortabilityError(f"Checksum or size mismatch: {relative}")
    except zipfile.BadZipFile as exc:
        raise PortabilityError("Not a valid ZIP archive") from exc
    return manifest


def import_archive(
    archive_path: str | Path,
    settings: Settings | None = None,
    *,
    overwrite: bool = False,
) -> dict:
    manifest = verify_archive(archive_path)
    settings = settings or get_settings()
    destination = settings.runs_dir
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(Path(archive_path).expanduser().resolve()) as archive:
        members = [info for info in archive.infolist() if info.filename != MANIFEST_NAME]
        targets = []
        for info in members:
            relative = _safe_archive_path(info.filename)
            target = (destination / Path(*relative.parts[1:])).resolve()
            if destination.resolve() not in target.parents:
                raise PortabilityError(f"Import path escapes configured runs directory: {info.filename}")
            if target.exists() and not overwrite:
                raise FileExistsError(f"Import would overwrite {target}; pass overwrite=True to replace files")
            targets.append((info, target))
        for info, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    return manifest
