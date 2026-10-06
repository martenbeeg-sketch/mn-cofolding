from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import streamlit as st

from .results import format_duration


DEFAULT_RESULTS_ROOT = Path("/mnt/data/RESULTS/protenix-optimization-playbook-20261001")
TARGETS = [
    ("CHRNA7", 502),
    ("SLC26A8", 970),
    ("NUP155", 1391),
    ("LCT", 1927),
    ("EP300", 2414),
    ("CEP350", 3117),
    ("DMD", 3685),
    ("PRKDC", 4128),
]
MODES = ("off", "fast", "big")
PREDICTION_FAILURES = {
    "oom", "timeout", "kernel_refusal", "invalid_input", "error",
    "missing_structure", "invalid_output", "partial_output",
}


def results_root() -> Path:
    value = os.environ.get("MN_PROTENIX_BENCHMARK_DIR")
    return Path(value).expanduser().resolve() if value else DEFAULT_RESULTS_ROOT


def load_attempts(root: Path | None = None) -> list[dict[str, Any]]:
    root = (root or results_root()).resolve()
    records_dir = root / "records"
    records = []
    if not records_dir.is_dir():
        return records
    for path in sorted(records_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(item, dict) and item.get("run_id"):
            records.append(item)
    return records


def _is_capacity_record(record: dict[str, Any]) -> bool:
    return "-capacity" in str(record.get("run_id", ""))


def _mode_records(records: list[dict[str, Any]], target: str, mode: str) -> list[dict[str, Any]]:
    rows = [
        row for row in records
        if row.get("target") == target and row.get("mode") == mode
        and (target == "CHRNA7" or _is_capacity_record(row))
    ]
    return sorted(rows, key=lambda row: str(row.get("ended_at") or row.get("started_at") or ""))


def _valid_pass(record: dict[str, Any]) -> bool:
    if record.get("status") != "pass":
        return False
    outputs = record.get("output_validation") or []
    return bool(outputs) and all(
        item.get("parse_exit_code") == 0 and item.get("confidence_json_valid") is True
        for item in outputs
    )


def _select_attempt(records: list[dict[str, Any]], target: str, mode: str) -> dict[str, Any] | None:
    rows = _mode_records(records, target, mode)
    passed = [row for row in rows if _valid_pass(row)]
    if passed:
        return passed[-1]
    prediction_failures = [row for row in rows if row.get("status") in PREDICTION_FAILURES]
    if prediction_failures:
        return prediction_failures[-1]
    other = [row for row in rows if row.get("status") not in {"launcher_error", "interrupted"}]
    if other:
        return other[-1]
    return rows[-1] if rows else None


def _attempt_cell(attempt: dict[str, Any] | None, *, pending: str = "Not run") -> str:
    if attempt is None:
        return html.escape(pending)
    run_id = str(attempt.get("run_id") or "")
    status = str(attempt.get("status") or "unknown")
    if _valid_pass(attempt):
        label = "PASS"
    elif status == "pass":
        label = "INVALID OUTPUT"
    else:
        label = status.replace("_", " ").upper()
    model_times = attempt.get("model_forward_seconds") or []
    model_seconds = max(model_times) if model_times else None
    wall_seconds = attempt.get("wall_seconds")
    peaks = (attempt.get("nvml_samples") or {}).values()
    peak_values = [row.get("peak_memory_mib") for row in peaks if isinstance(row.get("peak_memory_mib"), int)]
    details = []
    if model_seconds is not None:
        details.append(f"model {format_duration(model_seconds)}")
    if wall_seconds is not None:
        details.append(f"wall {format_duration(wall_seconds)}")
    if peak_values:
        details.append("peak " + "/".join(f"{value:,}" for value in peak_values) + " MiB")
    body = " · ".join([label, *details])
    href = "?protenix_attempt=" + quote(run_id, safe="")
    return f'<a href="{html.escape(href, quote=True)}">{html.escape(body)}</a>'


def protenix_benchmark_table(records: list[dict[str, Any]]) -> str:
    first_failure: dict[str, int] = {}
    for mode in MODES:
        for index, (target, _length) in enumerate(TARGETS):
            attempt = _select_attempt(records, target, mode)
            if attempt and attempt.get("status") in PREDICTION_FAILURES:
                first_failure[mode] = index
                break

    header = (
        "<table><thead><tr><th>Target</th><th>Length</th><th>off · 1 GPU</th>"
        "<th>fast · 1 GPU</th><th>big · 2 GPUs</th></tr></thead><tbody>"
    )
    rows = []
    for index, (target, length) in enumerate(TARGETS):
        cells = []
        for mode in MODES:
            attempt = _select_attempt(records, target, mode)
            pending = "Not run"
            if attempt is None and mode in first_failure and index > first_failure[mode]:
                pending = "Skipped after earlier failure"
            cells.append(f"<td>{_attempt_cell(attempt, pending=pending)}</td>")
        rows.append(
            "<tr>"
            f"<td>{html.escape(target)}</td><td>{length:,} aa</td>"
            + "".join(cells)
            + "</tr>"
        )
    return header + "".join(rows) + "</tbody></table>"


def _safe_path(raw_path: object, root: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _safe_dir_path(raw_path: object, root: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_dir() else None


def _confidence_metrics(record: dict[str, Any], root: Path) -> dict[str, Any]:
    metrics = {}
    for output in record.get("output_validation") or []:
        values = output.get("confidence_metrics") or {}
        confidence_path = _safe_path(output.get("confidence_json"), root)
        if not values and confidence_path:
            try:
                values = json.loads(confidence_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                values = {}
        if isinstance(values, dict):
            metrics.update({key: values[key] for key in ("plddt", "gpde", "ptm", "iptm", "ranking_score", "has_clash") if key in values})
    return metrics


@st.cache_data(show_spinner="Preparing structure viewer…", ttl=3600, max_entries=8)
def _structure_viewer(path_value: str, modified_ns: int) -> str:
    del modified_ns
    from py2Dmol import view

    viewer = view(size=(760, 760), controls=True, box=True, color="plddt", style="richardson", bg="white")
    viewer.add_pdb(path_value, name="prediction", use_biounit=False)
    return viewer.to_html(title="Protenix benchmark prediction")


def render_protenix_attempt(run_id: str, root: Path | None = None) -> None:
    root = (root or results_root()).resolve()
    record_path = None
    attempt = None
    for candidate_path in sorted((root / "records").glob("*.json")):
        try:
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(candidate, dict) and candidate.get("run_id") == run_id:
            record_path, attempt = candidate_path, candidate
            break
    st.markdown('[← Protenix benchmark](./?protenix_benchmark=1)')
    if attempt is None:
        st.error(f"No Protenix attempt found for `{run_id}`.")
        return

    target = str(attempt.get("target") or "Unknown target")
    mode = str(attempt.get("mode") or "unknown")
    status = "PASS" if _valid_pass(attempt) else str(attempt.get("status") or "unknown").replace("_", " ").upper()
    st.title(f"{target} · Protenix {mode}")
    st.caption(f"Attempt `{run_id}` · {attempt.get('image', 'image unknown')} · {attempt.get('upstream_pin', 'upstream pin unknown')}")

    related = [
        row for row in load_attempts(root)
        if row.get("target") == target and row.get("mode") == mode and row.get("run_id") != run_id
    ]
    if related:
        with st.expander(f"Other {mode} attempts for {target} ({len(related)})"):
            rows = []
            for row in related:
                other_id = str(row.get("run_id") or "")
                other_status = "PASS" if _valid_pass(row) else str(row.get("status") or "unknown").upper()
                href = "?protenix_attempt=" + quote(other_id, safe="")
                times = row.get("model_forward_seconds") or []
                inference = format_duration(max(times)) if times else "—"
                wall = format_duration(row.get("wall_seconds"))
                rows.append(
                    "<tr>"
                    f"<td><a href=\"{html.escape(href, quote=True)}\">{html.escape(other_id)}</a></td>"
                    f"<td>{html.escape(other_status)}</td><td>{html.escape(inference)}</td><td>{html.escape(wall)}</td>"
                    "</tr>"
                )
            st.markdown(
                "<table><thead><tr><th>Attempt</th><th>Status</th><th>Inference</th><th>Wall</th></tr></thead>"
                "<tbody>" + "".join(rows) + "</tbody></table>",
                unsafe_allow_html=True,
            )

    model_times = attempt.get("model_forward_seconds") or []
    metrics = st.columns(5)
    metrics[0].metric("Status", status)
    metrics[1].metric("Sequence", f"{attempt.get('target_length', '—')} aa")
    metrics[2].metric("Inference", format_duration(max(model_times)) if model_times else "—")
    metrics[3].metric("Wall time", format_duration(attempt.get("wall_seconds")))
    metrics[4].metric("GPU count", str(attempt.get("gpu_count", "—")))
    if attempt.get("prediction_failure") or attempt.get("interruption_reason"):
        st.warning(str(attempt.get("prediction_failure") or attempt.get("interruption_reason")))

    st.subheader("Input and run settings")
    input_path = _safe_path(attempt.get("input_json"), root)
    if input_path:
        st.code(input_path.read_text(errors="replace")[:30000], language="json")
        st.download_button(
            "Download input JSON",
            data=input_path.read_bytes(),
            file_name=input_path.name,
            mime="application/json",
            key=f"protenix-input-{run_id}",
        )
    if record_path:
        st.download_button(
            "Download structured attempt record",
            data=record_path.read_bytes(),
            file_name=record_path.name,
            mime="application/json",
            key=f"protenix-record-{run_id}",
        )
    st.json({
        "image": attempt.get("image"),
        "image_digest": attempt.get("image_digest"),
        "upstream_pin": attempt.get("upstream_pin"),
        "optimizer_kit_revision": attempt.get("optimizer_kit_revision"),
        "settings": attempt.get("effective_settings") or attempt.get("settings"),
        "input_sha256": attempt.get("input_sha256"),
        "sequence_sha256": attempt.get("sequence_sha256"),
        "msa_sha256": attempt.get("msa_sha256"),
        "msa_rows_available": attempt.get("msa_rows_available"),
        "msa_rows_used_per_prediction": attempt.get("msa_rows_used_per_prediction"),
        "gpu_assignment": attempt.get("gpu_assignment"),
    })

    st.subheader("Timing and GPU memory")
    phases = attempt.get("phase_timings") or {}
    phase_rows = [{"Phase": key.replace("_", " "), "Seconds": value} for key, value in phases.items() if isinstance(value, (int, float, list))]
    if phase_rows:
        st.dataframe(phase_rows, hide_index=True, width="stretch")
    gpu_rows = [
        {"GPU": gpu, **values}
        for gpu, values in (attempt.get("nvml_samples") or {}).items()
    ]
    if gpu_rows:
        st.dataframe(gpu_rows, hide_index=True, width="stretch")

    confidence = _confidence_metrics(attempt, root)
    if confidence:
        st.subheader("Protenix confidence summary")
        st.json(confidence)
    confidence_paths = sorted({
        path
        for output in (attempt.get("output_validation") or [])
        if (path := _safe_path(output.get("confidence_json"), root)) is not None
    }, key=str)
    for confidence_path in confidence_paths:
        st.download_button(
            "Download confidence JSON",
            data=confidence_path.read_bytes(),
            file_name=confidence_path.name,
            mime="application/json",
            key=f"protenix-confidence-{run_id}-{confidence_path.name}",
        )

    output_dir = _safe_dir_path(attempt.get("paths", {}).get("output"), root)
    if output_dir:
        structures = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".cif", ".mmcif", ".pdb"})
    else:
        structures = []
    if structures:
        structure = structures[0]
        st.subheader("Predicted structure")
        try:
            st.iframe(_structure_viewer(str(structure), structure.stat().st_mtime_ns), height=780, width="stretch")
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            st.warning(f"Interactive viewer unavailable: {exc}")
        st.download_button(
            "Download structure",
            data=structure.read_bytes(),
            file_name=structure.name,
            mime="chemical/x-cif" if structure.suffix.lower() in {".cif", ".mmcif"} else "chemical/x-pdb",
            key=f"protenix-structure-{run_id}",
        )
    else:
        st.info("No validated structure was written for this attempt.")

    st.subheader("Mode activation, command, and logs")
    st.caption("The final mode line records the parent decision, active levers, sharding, fallbacks, and partial status.")
    st.code("\n".join(attempt.get("active_mode_lines") or []), language="text")
    st.code("\n".join(attempt.get("final_mode_lines") or []), language="text")
    st.code(str(attempt.get("command_shell") or "Command not recorded."), language="bash")
    for label, path_key in (("Run log", "log"), ("NVML samples", "nvml")):
        path = _safe_path(attempt.get("paths", {}).get(path_key), root)
        if path:
            with st.expander(label):
                if path.suffix.lower() == ".csv":
                    st.download_button(f"Download {label.lower()}", path.read_bytes(), file_name=path.name, key=f"protenix-{path_key}-{run_id}")
                    st.code(path.read_text(errors="replace")[-20000:], language="text")
                else:
                    text = path.read_text(errors="replace")
                    st.download_button(f"Download {label.lower()}", text, file_name=path.name, key=f"protenix-{path_key}-{run_id}")
                    st.code(text[-40000:], language="text")
