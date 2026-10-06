from __future__ import annotations

from pathlib import Path

from mn_cofolding.protenix_benchmark import _safe_path, protenix_benchmark_table


def _record(target: str, mode: str, run_id: str, status: str) -> dict:
    record = {
        "target": target,
        "mode": mode,
        "run_id": run_id,
        "status": status,
        "wall_seconds": 120,
        "model_forward_seconds": [100],
        "gpu_count": 1,
        "nvml_samples": {"0": {"peak_memory_mib": 20000}},
        "output_validation": [],
    }
    if status == "pass":
        record["output_validation"] = [{"parse_exit_code": 0, "confidence_json_valid": True}]
    return record


def test_matrix_links_attempts_and_marks_larger_targets_skipped_after_failure() -> None:
    records = [
        _record("SLC26A8", "off", "off-SLC26A8-capacity-gpu0-1", "pass"),
        _record("NUP155", "off", "off-NUP155-capacity-gpu0-1", "oom"),
        _record("SLC26A8", "big", "big-SLC26A8-capacity-gpu0-1", "pass"),
    ]

    table = protenix_benchmark_table(records)

    assert "?protenix_attempt=off-SLC26A8-capacity-gpu0-1" in table
    assert "?protenix_attempt=big-SLC26A8-capacity-gpu0-1" in table
    assert "OOM" in table
    assert "Skipped after earlier failure" in table
    assert "big · 2 GPUs" in table


def test_unvalidated_pass_is_not_shown_as_pass() -> None:
    record = _record("SLC26A8", "fast", "fast-SLC26A8-capacity-gpu0-1", "pass")
    record["output_validation"] = []
    table = protenix_benchmark_table([record])

    assert "INVALID OUTPUT" in table
    assert "PASS" not in table


def test_safe_path_rejects_paths_outside_results_root(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    inside = root / "attempt.json"
    outside = tmp_path / "outside.json"
    inside.write_text("{}")
    outside.write_text("{}")

    assert _safe_path(str(inside), root) == inside
    assert _safe_path(str(outside), root) is None
