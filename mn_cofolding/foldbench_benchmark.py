from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any
from urllib.parse import quote

import streamlit as st


EXPERIMENT = "foldbench-heterodimer-length-2026-10-04"
DEFAULT_RESULTS_ROOT = Path(
    "/mnt/data/RESULTS/mn-cofolding-workdir/workdir/benchmarks/"
    "foldbench-heterodimer-length-20261004"
)
ENGINE_LABELS = {
    "alphafold3": "AlphaFold 3",
    "colabfold": "AlphaFold 2 Multimer · ColabFold kit",
    "boltz2": "Boltz-2",
    "protenix": "Protenix v2",
    "rf3_foundry": "RF3 Foundry",
    "af2ig": "AF2-IG · speed only",
    "esmfold2": "ESMFold 2 · optimization kit",
    "chai1": "Chai-1 · optimization kit",
}
ENGINE_ORDER = tuple(ENGINE_LABELS)
# Keep engine identity stable across every chart, including charts where one or
# more engines are absent from the selected data.
ENGINE_COLORS = {
    "alphafold3": "#4C78A8",
    "colabfold": "#F58518",
    "boltz2": "#54A24B",
    "protenix": "#B279A2",
    "rf3_foundry": "#E45756",
    "af2ig": "#72B7B2",
    "esmfold2": "#FF9DA6",
    "chai1": "#79706E",
}
CHART_DPI = 100
DEFAULT_CHART_LAYOUT = {
    "panel_width_px": 320,
    "panel_height_px": 180,
    "column_gap_px": 35,
    "row_gap_px": 12,
    "legend_height_px": 72,
    "side_margin_px": 80,
    "top_margin_px": 50,
    "bottom_margin_px": 10,
}
CHART_TITLE_WIDTH = 24
AXIS_LABEL_WIDTH = 20
MODE_ORDER = ("off", "exact", "fast", "big")
MODE_COLORS = {
    "off": "#4C78A8",
    "exact": "#F58518",
    "fast": "#54A24B",
    "big": "#E45756",
}


def results_root() -> Path:
    configured = os.environ.get("MN_FOLDBENCH_BENCHMARK_DIR")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_RESULTS_ROOT


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _load_af2ig_steady_profiles(
    root: Path, targets: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the matched AF2-IG timing repeats without importing any quality scores."""
    run_root = root / "af2ig" / "steady_state_20261006"
    batch_path = run_root / "benchmark_summary.json"
    if not batch_path.is_file():
        return [], {}
    try:
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [], {}
    if batch.get("status") != "completed":
        return [], {}

    target_index = {
        str(row.get("pdb_id")): row
        for row in targets
        if isinstance(row, dict) and row.get("pdb_id")
    }
    target_ids = list(target_index)
    mode_summaries: dict[str, dict[str, Any]] = {}
    per_mode_target: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in MODE_ORDER:
        path = run_root / "summaries" / f"{mode}.json"
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return [], {}
        timing_rows = summary.get("per_target_timing_summary", [])
        row_index = {
            str(row.get("target")): row
            for row in timing_rows
            if isinstance(row, dict) and row.get("target")
        }
        if (
            summary.get("exit_code") != 0
            or summary.get("design_records_found") != 20
            or any(
                target not in row_index
                or not row_index[target].get("timed_repeats_complete")
                or not isinstance(row_index[target].get("median_timed_forward_seconds"), (int, float))
                for target in target_ids
            )
        ):
            return [], {}
        mode_summaries[mode] = summary
        per_mode_target[mode] = row_index

    off_times = {
        target: float(per_mode_target["off"][target]["median_timed_forward_seconds"])
        for target in target_ids
    }
    profiles: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        for pdb_id in target_ids:
            timing = per_mode_target[mode][pdb_id]
            forward_seconds = float(timing["median_timed_forward_seconds"])
            target = target_index[pdb_id]
            profiles.append(
                {
                    "pdb_id": pdb_id,
                    "target_title": target.get("title", pdb_id),
                    "total_residues": int(target.get("total_residues") or 0),
                    "engine": "af2ig",
                    "engine_label": ENGINE_LABELS["af2ig"],
                    "mode": mode,
                    "run_status": "completed",
                    "reason": "Speed-only profile: median of three warmed repeats; DockQ omitted.",
                    "n_structures_found": 1,
                    "expected_structures": 1,
                    "n_scored": 0,
                    "mean_DockQ": None,
                    "wall_seconds": None,
                    "inference_seconds": forward_seconds,
                    "inference_seconds_by_seed": {
                        f"repeat_{index + 1}": float(value)
                        for index, value in enumerate(timing["timed_forward_seconds"])
                    },
                    "inference_speedup_vs_off": off_times[pdb_id] / forward_seconds,
                    "timing_repeats": 3,
                    "timing_scope": "median per-design model forward time after one in-process warm-up copy",
                    "gpu_ids": [0],
                    "peak_memory_mib_by_gpu": {},
                    "profile_complete": True,
                    "completed_seeds": 1,
                    "forward_display_seconds": forward_seconds,
                    "forward_per_seed_seconds": forward_seconds,
                    "gpu0_gib": None,
                    "gpu1_gib": None,
                    "app_status": "timing only",
                    "app_job_id": "",
                    "app_job_url": "",
                }
            )

    off_wall = float(mode_summaries["off"]["process_wall_seconds_including_kit_warmup_and_all_predictions"])
    batch_wall = {
        mode: float(summary["process_wall_seconds_including_kit_warmup_and_all_predictions"])
        for mode, summary in mode_summaries.items()
    }
    conditions = {
        "status": "completed",
        "run_directory": str(run_root),
        "timing_method": "One untimed input copy and three timed copies per target in one prediction process per mode; plotted forward values are per-target medians.",
        "timing_repeats_per_target": 3,
        "deterministic": True,
        "recycles": 3,
        "gpu_ids": [0],
        "msa": "Single-sequence features; no external MSA.",
        "input_condition": mode_summaries["off"].get("warmup_and_prediction_settings", {}).get("input_condition", ""),
        "dockq": "Excluded from FoldBench figures for AF2-IG.",
        "image": mode_summaries["off"].get("image"),
        "gpu_preset": mode_summaries["off"].get("gpu_config"),
        "batch_wall_seconds_including_mode_warmup": batch_wall,
        "batch_wall_speedup_vs_off": {
            mode: off_wall / seconds for mode, seconds in batch_wall.items() if seconds > 0
        },
    }
    return profiles, conditions


@st.cache_data(show_spinner=False, ttl=60, max_entries=4)
def _load_data(root_text: str, results_mtime: int, jobs_mtime: int, af2ig_mtime: int) -> dict[str, Any] | None:
    del results_mtime, jobs_mtime, af2ig_mtime  # Included in the cache key for file-change invalidation.
    root = Path(root_text)
    result_path = root / "benchmark_results.json"
    if not result_path.is_file():
        return None
    try:
        results = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(results, dict) or not isinstance(results.get("profile_summaries"), list):
        return None

    jobs_path = root / "registered_jobs.json"
    try:
        registered = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.is_file() else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        registered = {}
    jobs = registered.get("jobs", []) if isinstance(registered, dict) else []
    job_index = {
        (str(row.get("target")), str(row.get("engine")), str(row.get("mode"))): row
        for row in jobs
        if isinstance(row, dict)
    }

    targets = results.get("targets", [])
    target_index = {
        str(row.get("pdb_id")): row
        for row in targets
        if isinstance(row, dict) and row.get("pdb_id")
    }
    profiles = []
    for item in results["profile_summaries"]:
        if not isinstance(item, dict):
            continue
        pdb_id = str(item.get("pdb_id") or "")
        engine = str(item.get("engine") or "")
        mode = str(item.get("mode") or "")
        target = target_index.get(pdb_id, {})
        job = job_index.get((pdb_id, engine, mode), {})
        structures = int(item.get("n_structures_found") or 0)
        scored = int(item.get("n_scored") or 0)
        expected_structures = int(item.get("expected_structures") or 15)
        full = (
            item.get("run_status") == "completed"
            and structures == expected_structures
            and scored == expected_structures
        )
        if engine in {"esmfold2", "chai1"}:
            engine_settings = results.get("conditions", {}).get(f"{engine}_settings", {})
            samples_per_seed = max(1, int(engine_settings.get("samples_per_seed") or 1))
            expected_seeds = len(engine_settings.get("seeds") or [])
            completed_seeds = structures // samples_per_seed
            if expected_seeds:
                completed_seeds = min(expected_seeds, completed_seeds)
        else:
            completed_seeds = min(3, structures // 5) if expected_structures == 15 else structures
        inference_seconds = item.get("inference_seconds")
        forward_display_seconds = (
            float(inference_seconds) / completed_seeds
            if isinstance(inference_seconds, (int, float)) and completed_seeds > 0
            else None
        )
        memory = item.get("peak_memory_mib_by_gpu") or {}
        profiles.append(
            {
                **item,
                "pdb_id": pdb_id,
                "target_title": target.get("title", pdb_id),
                "total_residues": int(item.get("total_residues") or target.get("total_residues") or 0),
                "engine_label": ENGINE_LABELS.get(engine, engine),
                "profile_complete": full,
                "completed_seeds": completed_seeds,
                "expected_structures": expected_structures,
                "forward_display_seconds": forward_display_seconds,
                "forward_per_seed_seconds": forward_display_seconds,
                "gpu0_gib": float(memory["0"]) / 1024 if memory.get("0") is not None else None,
                "gpu1_gib": float(memory["1"]) / 1024 if memory.get("1") is not None else None,
                "app_status": item.get("app_status") or job.get("status", "not registered"),
                "app_job_id": job.get("job_id", ""),
                "app_job_url": f"?job_id={quote(str(job.get('job_id') or ''), safe='')}" if job.get("job_id") else "",
            }
        )

    af2ig_profiles, af2ig_conditions = _load_af2ig_steady_profiles(root, targets)
    profiles.extend(af2ig_profiles)
    conditions = dict(results.get("conditions", {}))
    if af2ig_conditions:
        conditions["af2ig"] = af2ig_conditions

    return {
        "root": root_text,
        "experiment": results.get("experiment", EXPERIMENT),
        "conditions": conditions,
        "targets": targets,
        "profiles": profiles,
        "structures": results.get("structures", []),
        "registered_jobs": jobs,
        "profile_csv": root / "profile_summary.csv",
        "af2ig_profile_csv": root / "af2ig" / "steady_state_20261006" / "af2ig_steady_state_profile.csv",
        "structure_csv": root / "structure_scores.csv",
        "results_json": result_path,
    }


def load_foldbench_data(root: Path | None = None) -> dict[str, Any] | None:
    root = (root or results_root()).expanduser().resolve()
    af2ig_root = root / "af2ig" / "steady_state_20261006"
    af2ig_mtime = max(
        (
            _mtime_ns(path)
            for path in [
                af2ig_root / "benchmark_summary.json",
                af2ig_root / "af2ig_steady_state_profile.csv",
                *sorted((af2ig_root / "summaries").glob("*.json")),
            ]
        ),
        default=0,
    )
    return _load_data(
        str(root),
        _mtime_ns(root / "benchmark_results.json"),
        _mtime_ns(root / "registered_jobs.json"),
        af2ig_mtime,
    )


def _chart_layout_controls() -> dict[str, int]:
    """Expose the chart canvas geometry as live dashboard controls."""
    with st.expander("Chart layout controls", expanded=False):
        st.caption(
            "Enter exact pixel values. Width and height change every panel; the gaps "
            "control the distance between columns and panel rows. Increase the side "
            "margin if a long axis title needs more room."
        )
        controls = st.columns(6)
        with controls[0]:
            panel_width = st.number_input(
                "Panel width (px)",
                min_value=240,
                value=DEFAULT_CHART_LAYOUT["panel_width_px"],
                step=10,
                format="%d",
                key="foldbench_panel_width_px",
                help="Increase this when labels or curves feel cramped horizontally.",
            )
        with controls[1]:
            panel_height = st.number_input(
                "Panel height (px)",
                min_value=140,
                value=DEFAULT_CHART_LAYOUT["panel_height_px"],
                step=10,
                format="%d",
                key="foldbench_panel_height_px",
                help="Increase this when the curves or y-axis labels feel cramped vertically.",
            )
        with controls[2]:
            column_gap = st.number_input(
                "Column gap (px)",
                min_value=0,
                value=DEFAULT_CHART_LAYOUT["column_gap_px"],
                step=5,
                format="%d",
                key="foldbench_column_gap_px",
                help="Horizontal distance between panels in the same row.",
            )
        with controls[3]:
            row_gap = st.number_input(
                "Row gap (px)",
                min_value=0,
                value=DEFAULT_CHART_LAYOUT["row_gap_px"],
                step=5,
                format="%d",
                key="foldbench_row_gap_px",
                help="Extra vertical distance between one panel row and the next.",
            )
        with controls[4]:
            legend_height = st.number_input(
                "Legend space (px)",
                min_value=40,
                value=DEFAULT_CHART_LAYOUT["legend_height_px"],
                step=5,
                format="%d",
                key="foldbench_legend_height_px",
                help="Space reserved below each panel row for its curve legend.",
            )
        with controls[5]:
            side_margin = st.number_input(
                "Side margin (px)",
                min_value=30,
                value=DEFAULT_CHART_LAYOUT["side_margin_px"],
                step=10,
                format="%d",
                key="foldbench_side_margin_px",
                help="Extra left/right room for long axis titles and tick labels.",
            )
    return {
        **DEFAULT_CHART_LAYOUT,
        "panel_width_px": int(panel_width),
        "panel_height_px": int(panel_height),
        "column_gap_px": int(column_gap),
        "row_gap_px": int(row_gap),
        "legend_height_px": int(legend_height),
        "side_margin_px": int(side_margin),
    }


def _complete_profiles(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in data["profiles"] if row.get("profile_complete")]


def _paired_summary(data: dict[str, Any], engine: str, mode: str) -> tuple[int, float | None, float | None]:
    rows = {
        str(row["pdb_id"]): row
        for row in _complete_profiles(data)
        if row.get("engine") == engine and row.get("mode") == mode
    }
    off = {
        str(row["pdb_id"]): row
        for row in _complete_profiles(data)
        if row.get("engine") == engine and row.get("mode") == "off"
    }
    paired = [(row, off[pdb]) for pdb, row in rows.items() if pdb in off]
    speedups = [
        float(base["inference_seconds"]) / float(row["inference_seconds"])
        for row, base in paired
        if row.get("inference_seconds") and base.get("inference_seconds")
    ]
    dockq_deltas = [
        float(row["mean_DockQ"]) - float(base["mean_DockQ"])
        for row, base in paired
        if row.get("mean_DockQ") is not None and base.get("mean_DockQ") is not None
    ]
    if engine == "af2ig" and paired:
        speedup_summary = (
            sum(float(base["inference_seconds"]) for _row, base in paired)
            / sum(float(row["inference_seconds"]) for row, _base in paired)
        )
    else:
        speedup_summary = sum(speedups) / len(speedups) if speedups else None
    return (
        len(paired),
        speedup_summary,
        sum(dockq_deltas) / len(dockq_deltas) if dockq_deltas else None,
    )


def render_foldbench_overview(data: dict[str, Any] | None = None) -> None:
    data = data or load_foldbench_data()
    if data is None:
        return

    complete = _complete_profiles(data)
    scored_count = len(data.get("structures", []))
    total_profiles = len(data.get("profiles", []))
    st.divider()
    st.subheader("FoldBench heterodimer benchmark")
    st.caption(
        "Sequence engines use shared local unpaired MSAs. Most use three seeds × five structures; ESMFold 2 "
        "and Chai-1 use 15 seeds × one structure. AF2-IG contributes speed-only profiles from three warm repeats; "
        "it starts from the supplied native complex and is excluded from DockQ figures. "
        "Other predictions are scored with DockQ-v2 against the native chain pair. "
        "Open the dashboard for target-length plots, GPU memory, and per-structure scores."
    )
    cols = st.columns(4)
    cols[0].metric("Complete profiles", f"{len(complete)}/{total_profiles}")
    cols[1].metric("DockQ scores", f"{scored_count:,}")
    cols[2].metric("Targets", str(len(data.get("targets", []))))
    cols[3].metric("Registered app jobs", str(len(data.get("registered_jobs", []))))

    rows = []
    for engine in ENGINE_ORDER:
        fast_n, fast_speed, fast_delta = _paired_summary(data, engine, "fast")
        _big_n, big_speed, _big_delta = _paired_summary(data, engine, "big")
        rows.append(
            {
                "Engine": ENGINE_LABELS[engine],
                "Fast complete targets": f"{fast_n}/5",
                "Fast forward speedup": f"{fast_speed:.2f}×" if fast_speed is not None else "—",
                "Paired DockQ change": f"{fast_delta:+.3f}" if fast_delta is not None else "—",
                "Big forward speedup": f"{big_speed:.2f}×" if big_speed is not None else "—",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.markdown("[Open the interactive FoldBench dashboard](?foldbench_benchmark=1)")


def _selected_profiles(
    data: dict[str, Any],
    *,
    modes: list[str] | None,
    engines: list[str],
    targets: list[str],
) -> list[dict[str, Any]]:
    return [
        row for row in _complete_profiles(data)
        if (modes is None or row.get("mode") in modes)
        and row.get("engine") in engines
        and row.get("pdb_id") in targets
    ]


def _render_chart(fig: Any | None, *, centered: bool = False) -> None:
    """Render a chart at dashboard scale and close its Matplotlib figure."""
    if fig is None:
        return

    from io import BytesIO

    target = st
    if centered:
        # A single chart should read as a compact panel instead of expanding
        # across the entire wide dashboard.
        target = st.columns([1, 2, 1])[1]

    # st.pyplot applies its own tight bounding box and DPI and warns when
    # those save options are overridden.  Serialize the figure ourselves so
    # every chart keeps the exact canvas dimensions created above, regardless
    # of how wide its labels happen to be.
    image = BytesIO()
    fig.savefig(image, format="png", dpi=CHART_DPI, bbox_inches=None)
    image.seek(0)
    target.image(image, width="content")

    import matplotlib.pyplot as plt

    plt.close(fig)


def _wrap_chart_text(text: str, *, width: int) -> str:
    """Wrap chart text without splitting identifiers or long words."""
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _curve_legend_handles(
    rows: list[dict[str, Any]],
    *,
    series_by: str,
    threshold: float | None = None,
    threshold_label: str | None = None,
    include_gpu: bool = False,
) -> list[Any]:
    """Build legend handles for the curves shown inside each facet row."""
    from matplotlib.lines import Line2D

    if series_by == "engine":
        series_order = MODE_ORDER
        series_key = "mode"
        series_colors = MODE_COLORS
        series_labels = {mode: mode.title() for mode in MODE_ORDER}
    else:
        series_order = ENGINE_ORDER
        series_key = "engine"
        series_colors = ENGINE_COLORS
        series_labels = ENGINE_LABELS

    handles = []
    for series_name in series_order:
        if any(row.get(series_key) == series_name for row in rows):
            handles.append(
                Line2D(
                    [],
                    [],
                    color=series_colors.get(series_name, "#4C4C4C"),
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    label=series_labels[series_name],
                )
            )
    if include_gpu:
        handles.extend(
            [
                Line2D([], [], color="#4C4C4C", marker="o", linestyle="", label="GPU 0"),
                Line2D([], [], color="#4C4C4C", marker="s", linestyle="", label="GPU 1"),
            ]
        )
    if threshold is not None:
        handles.append(
            Line2D(
                [],
                [],
                color="#777777",
                linestyle="--",
                linewidth=1,
                label=threshold_label or f"Threshold {threshold:.2f}",
            )
        )
    return handles


def _make_facet_figure(
    panel_count: int,
    *,
    series_by: str,
    title: str,
    layout: dict[str, int] | None = None,
) -> tuple[Any, list[list[Any]], list[Any]]:
    """Create pixel-sized facet axes with a legend row beneath every panel row."""
    import matplotlib.pyplot as plt

    layout = {**DEFAULT_CHART_LAYOUT, **(layout or {})}
    ncols = 1 if series_by == "mode" else (2 if panel_count > 1 else 1)
    panel_rows = max(1, (panel_count + ncols - 1) // ncols)
    panel_width = layout["panel_width_px"]
    panel_height = layout["panel_height_px"]
    column_gap = layout["column_gap_px"]
    row_gap = layout["row_gap_px"]
    legend_height = layout["legend_height_px"]
    side_margin = layout["side_margin_px"]
    top_margin = layout["top_margin_px"]
    bottom_margin = layout["bottom_margin_px"]
    figure_width = 2 * side_margin + ncols * panel_width + max(0, ncols - 1) * column_gap
    row_height = panel_height + legend_height
    figure_height = (
        top_margin
        + bottom_margin
        + panel_rows * row_height
        + max(0, panel_rows - 1) * row_gap
    )
    fig = plt.figure(
        figsize=(figure_width / CHART_DPI, figure_height / CHART_DPI),
        dpi=CHART_DPI,
    )
    panel_axes = []
    legend_axes = []
    panel_index = 0
    for row_index in range(panel_rows):
        row_top = top_margin + row_index * (row_height + row_gap)
        panel_bottom = figure_height - row_top - panel_height
        row_axes = []
        for col_index in range(ncols):
            panel_left = side_margin + col_index * (panel_width + column_gap)
            axis = fig.add_axes(
                [
                    panel_left / figure_width,
                    panel_bottom / figure_height,
                    panel_width / figure_width,
                    panel_height / figure_height,
                ]
            )
            if panel_index >= panel_count:
                axis.set_visible(False)
            row_axes.append(axis)
            panel_index += 1
        legend_bottom = panel_bottom - legend_height
        legend_axis = fig.add_axes(
            [
                side_margin / figure_width,
                legend_bottom / figure_height,
                (figure_width - 2 * side_margin) / figure_width,
                legend_height / figure_height,
            ]
        )
        legend_axis.set_axis_off()
        panel_axes.append(row_axes)
        legend_axes.append(legend_axis)
    fig.suptitle(title, y=1 - max(8, top_margin * 0.35) / figure_height, fontsize=10)
    return fig, panel_axes, legend_axes


def _facet_spec(rows: list[dict[str, Any]], *, series_by: str) -> tuple[list[str], str, list[str], str]:
    """Return panel order/key and line order/key for the selected visualization."""
    if series_by == "mode":
        return (
            [mode for mode in MODE_ORDER if any(row.get("mode") == mode for row in rows)],
            "mode",
            [engine for engine in ENGINE_ORDER if any(row.get("engine") == engine for row in rows)],
            "engine",
        )
    return (
        [engine for engine in ENGINE_ORDER if any(row.get("engine") == engine for row in rows)],
        "engine",
        [mode for mode in MODE_ORDER if any(row.get("mode") == mode for row in rows)],
            "mode",
        )


def _plot_line(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    threshold: float | None = None,
    threshold_label: str | None = None,
    series_by: str = "engine",
    layout: dict[str, int] | None = None,
) -> Any | None:
    valid = [row for row in rows if isinstance(row.get(metric), (int, float))]
    if not valid:
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    panel_order, panel_key, line_order, line_key = _facet_spec(valid, series_by=series_by)
    if not panel_order:
        return None
    fig, panel_axes, legend_axes = _make_facet_figure(
        len(panel_order),
        series_by=series_by,
        title=title,
        layout=layout,
    )
    for index, panel_name in enumerate(panel_order):
        ax = panel_axes[index // len(panel_axes[0])][index % len(panel_axes[0])]
        for line_name in line_order:
            series = sorted(
                (
                    row for row in valid
                    if row.get(panel_key) == panel_name and row.get(line_key) == line_name
                ),
                key=lambda row: int(row.get("total_residues") or 0),
            )
            if not series:
                continue
            engine = panel_name if series_by == "engine" else line_name
            mode = line_name if series_by == "engine" else panel_name
            ax.plot(
                [row["total_residues"] for row in series],
                [row[metric] for row in series],
                marker="o",
                linewidth=2,
                linestyle="-",
                color=(
                    MODE_COLORS.get(mode, "#4C4C4C")
                    if series_by == "engine"
                    else ENGINE_COLORS.get(engine, "#4C4C4C")
                ),
            )
        if threshold is not None:
            ax.axhline(
                threshold,
                color="#777777",
                linestyle="--",
                linewidth=1,
            )
        panel_label = ENGINE_LABELS[panel_name] if panel_key == "engine" else panel_name.title()
        ax.set_title(_wrap_chart_text(panel_label, width=CHART_TITLE_WIDTH), fontsize=9, pad=5)
        ax.set_xlabel("Total sequence length (aa)")
        ax.set_ylabel(_wrap_chart_text(ylabel, width=AXIS_LABEL_WIDTH))
        ax.grid(True, alpha=0.22)
    handles = _curve_legend_handles(
        valid,
        series_by=series_by,
        threshold=threshold,
        threshold_label=threshold_label,
    )
    if handles:
        for legend_axis in legend_axes:
            legend_axis.legend(
                handles=handles,
                loc="center",
                bbox_to_anchor=(0.5, 0.25),
                ncol=min(4, len(handles)),
                frameon=False,
                fontsize=7,
            )
    return fig


def _plot_memory(
    rows: list[dict[str, Any]],
    *,
    series_by: str = "engine",
    title: str = "Peak device memory by target",
    layout: dict[str, int] | None = None,
) -> Any | None:
    valid = [row for row in rows if row.get("gpu0_gib") is not None or row.get("gpu1_gib") is not None]
    if not valid:
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    panel_order, panel_key, line_order, line_key = _facet_spec(valid, series_by=series_by)
    if not panel_order:
        return None
    fig, panel_axes, legend_axes = _make_facet_figure(
        len(panel_order),
        series_by=series_by,
        title=title,
        layout=layout,
    )
    for index, panel_name in enumerate(panel_order):
        ax = panel_axes[index // len(panel_axes[0])][index % len(panel_axes[0])]
        for line_name in line_order:
            engine = panel_name if series_by == "engine" else line_name
            mode = line_name if series_by == "engine" else panel_name
            for gpu, key, marker in ((0, "gpu0_gib", "o"), (1, "gpu1_gib", "s")):
                series = sorted(
                    (
                        row for row in valid
                        if row.get(panel_key) == panel_name
                        and row.get(line_key) == line_name
                        and row.get(key) is not None
                    ),
                    key=lambda row: int(row.get("total_residues") or 0),
                )
                if not series:
                    continue
                ax.plot(
                    [row["total_residues"] for row in series],
                    [row[key] for row in series],
                    marker=marker,
                    linestyle="-",
                    linewidth=1.8,
                    color=(
                        MODE_COLORS.get(mode, "#4C4C4C")
                        if series_by == "engine"
                        else ENGINE_COLORS.get(engine, "#4C4C4C")
                    ),
                )
        panel_label = ENGINE_LABELS[panel_name] if panel_key == "engine" else panel_name.title()
        ax.set_title(_wrap_chart_text(panel_label, width=CHART_TITLE_WIDTH), fontsize=9, pad=5)
        ax.set_xlabel("Total sequence length (aa)")
        ax.set_ylabel(_wrap_chart_text("Peak memory (GiB)", width=AXIS_LABEL_WIDTH))
        ax.grid(True, alpha=0.22)
    handles = _curve_legend_handles(valid, series_by=series_by, include_gpu=True)
    if handles:
        for legend_axis in legend_axes:
            legend_axis.legend(
                handles=handles,
                loc="center",
                bbox_to_anchor=(0.5, 0.25),
                ncol=min(4, len(handles)),
                frameon=False,
                fontsize=7,
            )
    return fig


def _profile_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "Target": row.get("pdb_id"),
                "aa": row.get("total_residues"),
                "Engine": row.get("engine_label"),
                "Mode": row.get("mode"),
                "MSA rows (A / B)": " / ".join(
                    f"{int(value):,}" for value in (row.get("msa_rows_by_chain") or [])
                ),
                "ColabFold MSA cap": (
                    f"{int(row['colabfold_max_seq']):,} clustered + {int(row['colabfold_max_extra_seq']):,} extra"
                    if isinstance(row.get("colabfold_max_seq"), (int, float))
                    and isinstance(row.get("colabfold_max_extra_seq"), (int, float)) else "—"
                ),
                "Recycles observed (min / mean / max)": (
                    f"{int(row['recycles_observed_min'])} / {row['recycles_observed_mean']:.1f} / {int(row['recycles_observed_max'])}"
                    if all(isinstance(row.get(key), (int, float)) for key in (
                        "recycles_observed_min", "recycles_observed_mean", "recycles_observed_max"
                    )) else "—"
                ),
                "Run status": row.get("run_status"),
                "Timing repeats": row.get("timing_repeats") or "—",
                "Mode note": (
                    f"Levers off: {row.get('mode_levers_off')}"
                    if row.get("mode_levers_off")
                    else row.get("reason") or ""
                ),
                "App status": row.get("app_status"),
                "Structures": f"{row.get('n_structures_found', 0)}/{row.get('expected_structures', 15)}",
                "Wall (s)": row.get("wall_seconds"),
                "Forward (s)": row.get("forward_display_seconds"),
                "Mean DockQ": row.get("mean_DockQ"),
                "Successes": f"{row.get('successes', 0)}/{row.get('n_scored', 0)}",
                "GPUs": ", ".join(str(value) for value in row.get("gpu_ids", [])),
                "GPU 0 peak (GiB)": row.get("gpu0_gib"),
                "GPU 1 peak (GiB)": row.get("gpu1_gib"),
                "Result": row.get("app_job_url", ""),
                "Job ID": row.get("app_job_id", ""),
                "Complete": bool(row.get("profile_complete")),
            }
        )
    return output


def render_foldbench_dashboard(data: dict[str, Any] | None = None) -> None:
    data = data or load_foldbench_data()
    st.markdown("[← MN Cofolding](./)")
    if data is None:
        st.title("FoldBench heterodimer benchmark")
        st.error(
            f"Benchmark results were not found under `{results_root()}`. "
            "Set `MN_FOLDBENCH_BENCHMARK_DIR` to the folder containing `benchmark_results.json`."
        )
        return

    profiles = data["profiles"]
    complete = _complete_profiles(data)
    scored = data.get("structures", [])
    total_profiles = len(profiles)
    st.title("FoldBench heterodimer benchmark")
    st.caption(
        f"Experiment `{data.get('experiment', EXPERIMENT)}` · five two-chain targets. Most sequence-engine profiles use "
        "three seeds × five structures; ESMFold 2 and Chai-1 use 15 seeds × one structure. AF2-IG contributes "
        "speed-only profiles from three warm repeats per target/mode. All runs use GPU 0; big uses GPUs 0 and 1 for engines with "
        "two-GPU sharding, while Chai-1 big remains single-GPU. ColabFold uses its kit's AlphaFold 2 Multimer v3 workflow."
    )
    af2ig_conditions = data.get("conditions", {}).get("af2ig", {})
    if af2ig_conditions:
        st.info(
            "AF2-IG appears in speed figures only. It uses the native complex as structural input (target context plus binder initial guess) "
            "and single-sequence features without an external MSA. Its speed timings describe structure-informed refinement, "
            "so absolute runtimes are not directly comparable to sequence/MSA folding; DockQ is intentionally omitted."
        )
    st.caption(
        data.get("conditions", {}).get("colabfold_recycle_note", "")
        + " ColabFold logged max_seq=508 and max_extra_seq=2,048 for Multimer v3 MSA processing."
        if any(row.get("engine") == "colabfold" for row in profiles) else ""
    )
    summary_cols = st.columns(4)
    summary_cols[0].metric("Complete profiles", f"{len(complete)}/{total_profiles}")
    summary_cols[1].metric("Scored structures", f"{len(scored):,}")
    summary_cols[2].metric("Targets", str(len(data.get("targets", []))))
    summary_cols[3].metric("App jobs", str(len(data.get("registered_jobs", []))))

    target_order = [str(row.get("pdb_id")) for row in data.get("targets", []) if row.get("pdb_id")]
    target_index = {
        str(row.get("pdb_id")): row
        for row in data.get("targets", [])
        if row.get("pdb_id")
    }
    engine_options = [engine for engine in ENGINE_ORDER if any(row.get("engine") == engine for row in profiles)]
    filter_cols = st.columns([1.1, 2.5, 2.0, 3.4])
    with filter_cols[0]:
        chart_view = st.selectbox(
            "Visualization",
            options=["engine", "mode"],
            format_func=lambda value: "Engine view" if value == "engine" else "Mode view",
            key="foldbench_visualization",
        )
    with filter_cols[1]:
        selected_engines = st.multiselect(
            "Engines",
            options=engine_options,
            default=engine_options if chart_view == "engine" else engine_options[:1],
            format_func=lambda value: ENGINE_LABELS.get(value, value),
            key=("foldbench_engine_view_engines" if chart_view == "engine" else "foldbench_mode_view_engines"),
        )
    with filter_cols[2]:
        selected_modes = st.multiselect(
            "Modes",
            options=list(MODE_ORDER),
            default=(list(MODE_ORDER) if chart_view == "mode" else ["fast"]),
            format_func=lambda value: value.title(),
            key=("foldbench_engine_view_modes" if chart_view == "engine" else "foldbench_mode_view_modes"),
        )
    with filter_cols[3]:
        selected_targets = st.multiselect(
            "Targets",
            options=target_order,
            default=target_order,
            format_func=lambda value: f"{value} · {target_index[value].get('title', value)}",
            key="foldbench_targets",
        )

    chart_layout = _chart_layout_controls()
    chart_rows = _selected_profiles(
        data,
        modes=selected_modes,
        engines=selected_engines,
        targets=selected_targets,
    )
    if chart_view == "engine":
        st.caption("Engine view: each panel is one engine; selected modes use different colors inside that panel.")
    else:
        st.caption("Mode view: each panel is one mode; selected engines keep their fixed colors inside that panel.")
    if not chart_rows:
        st.warning("No complete profiles match the selected visualization filters.")
    else:
        st.caption(
            "Curve legends are placed below each panel row. Gray dashed lines mark the Speed off baseline (1.00×) "
            "or DockQ cutoff (0.23); memory markers: circle = GPU 0, square = GPU 1."
        )

    speed_tab, quality_tab, memory_tab, profiles_tab = st.tabs(
        ["Speed", "DockQ quality", "GPU memory", "All profiles"]
    )
    with speed_tab:
        speed_cols = st.columns(2)
        with speed_cols[0]:
            fig = _plot_line(
                chart_rows,
                metric="forward_display_seconds",
                title="Model forward time",
                ylabel="Forward seconds (per seed or single prediction)",
                series_by=chart_view,
                layout=chart_layout,
            )
            if fig is not None:
                _render_chart(fig)
        with speed_cols[1]:
            fig = _plot_line(
                chart_rows,
                metric="inference_speedup_vs_off",
                title="Speedup vs off",
                ylabel="Forward speedup vs off (×)",
                threshold=1.0,
                threshold_label="Off baseline (1.00×)",
                series_by=chart_view,
                layout=chart_layout,
            )
            if fig is not None:
                _render_chart(fig)
        st.caption(
            "Speedup is the off-mode forward time divided by each selected mode's forward time for the same target and engine. "
            "Values above 1× are faster than off; values below 1× are slower."
        )
        st.caption(
            "Forward time averages model calls over completed seeds: most engines use five structures per seed, while ESMFold 2 "
            "and Chai-1 use one. AF2-IG reports the per-target median of three warmed model-forward repeats. Its per-target "
            "wall time and GPU memory are omitted because the repeat run only supports matched forward timing. Shared MSA generation "
            "is excluded. Incomplete profiles are omitted."
        )
        af2ig_timing = data.get("conditions", {}).get("af2ig", {})
        batch_walls = af2ig_timing.get("batch_wall_seconds_including_mode_warmup", {})
        batch_speedups = af2ig_timing.get("batch_wall_speedup_vs_off", {})
        if batch_walls:
            st.subheader("AF2-IG full-run elapsed time")
            st.dataframe(
                [
                    {
                        "Mode": mode.title(),
                        "Full run wall (min)": f"{batch_walls[mode] / 60:.1f}",
                        "Wall speedup vs off": (
                            f"{batch_speedups[mode]:.2f}×" if batch_speedups.get(mode) is not None else "—"
                        ),
                    }
                    for mode in MODE_ORDER if mode in batch_walls
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "These are observed 20-copy batch times including each mode's kit warm-up. Off used the public warm-up; "
                "optimized modes warmed the five target lengths. The line chart above shows steady-state per-target forward speed."
            )

    with quality_tab:
        fig = _plot_line(
            chart_rows,
            metric="mean_DockQ",
            title="Mean DockQ by target",
            ylabel="Mean DockQ-v2",
            threshold=0.23,
            series_by=chart_view,
            layout=chart_layout,
        )
        if fig is not None:
            _render_chart(fig, centered=True)
        selected_scores = [
            {
                "Target": row.get("pdb_id"),
                "Engine": ENGINE_LABELS.get(str(row.get("engine")), row.get("engine")),
                "Mode": row.get("mode"),
                "Seed": row.get("seed"),
                "Sample": row.get("sample"),
                "DockQ": row.get("DockQ"),
                "iRMSD": row.get("iRMSD"),
                "LRMSD": row.get("LRMSD"),
                "Success ≥ 0.23": row.get("success"),
            }
            for row in scored
            if row.get("engine") in selected_engines
            and row.get("mode") in selected_modes
            and row.get("pdb_id") in selected_targets
        ]
        if selected_scores:
            st.subheader("Per-structure scores")
            st.dataframe(selected_scores, hide_index=True, width="stretch")
        st.caption(
            "The 0.23 line is the configured DockQ success cutoff. AF2-IG is intentionally omitted from DockQ figures; its "
            "structure-informed, single-sequence timing profile is shown only in the Speed tab."
        )

    with memory_tab:
        fig = _plot_memory(
            chart_rows,
            series_by=chart_view,
            title="Peak device memory by target",
            layout=chart_layout,
        )
        if fig is not None:
            _render_chart(fig, centered=True)
        st.caption(
            "Peak device-wide memory was sampled from nvidia-smi every 0.5 seconds. AF3 uses JAX memory reservation, "
            "so its reported device usage includes allocator reservation. Engine view uses engine colors; Mode view uses mode "
            "colors. Circle/square markers identify GPU 0/1."
        )
        memory_rows = [
            {
                "Target": row["pdb_id"],
                "aa": row["total_residues"],
                "Engine": row["engine_label"],
                "Mode": row["mode"],
                "GPU 0 peak (GiB)": row["gpu0_gib"],
                "GPU 1 peak (GiB)": row["gpu1_gib"],
            }
            for row in chart_rows
        ]
        if memory_rows:
            st.dataframe(memory_rows, hide_index=True, width="stretch")

    with profiles_tab:
        selected_rows = [
            row for row in profiles
            if row.get("engine") in selected_engines and row.get("pdb_id") in selected_targets
        ]
        table = _profile_table(selected_rows)
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Result": st.column_config.LinkColumn("MN Cofolding result", display_text="Open"),
                "Wall (s)": st.column_config.NumberColumn(format="%.1f"),
                "Forward (s)": st.column_config.NumberColumn(format="%.1f"),
                "Mean DockQ": st.column_config.NumberColumn(format="%.3f"),
                "GPU 0 peak (GiB)": st.column_config.NumberColumn(format="%.1f"),
                "GPU 1 peak (GiB)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        incomplete = [row for row in selected_rows if not row.get("profile_complete")]
        if incomplete:
            st.warning(
                "Incomplete rows remain visible for diagnosis but are excluded from speed and quality charts. "
                "See each row's status and mode note for details."
            )
        download_cols = st.columns(4)
        for col, key, label, mime in (
            (download_cols[0], "profile_csv", "Download profile CSV", "text/csv"),
            (download_cols[1], "af2ig_profile_csv", "Download AF2-IG timing CSV", "text/csv"),
            (download_cols[2], "structure_csv", "Download DockQ CSV", "text/csv"),
            (download_cols[3], "results_json", "Download benchmark JSON", "application/json"),
        ):
            path = data[key]
            if path.is_file():
                col.download_button(
                    label,
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    key=f"foldbench-download-{path.name}",
                )

    with st.expander("Benchmark conditions and target MSAs"):
        conditions = data.get("conditions", {})
        target_rows = [
            {
                "PDB": target.get("pdb_id"),
                "Target": target.get("title"),
                "Chains": ":".join(map(str, target.get("sequence_lengths", []))),
                "Total aa": target.get("total_residues"),
                "Notes": target.get("note") or "",
            }
            for target in data.get("targets", [])
        ]
        if target_rows:
            st.dataframe(target_rows, hide_index=True, width="stretch")
        st.json(
            {
                "standard sequence-engine settings": {
                    "seeds": conditions.get("seeds"),
                    "samples_per_seed": conditions.get("samples_per_seed"),
                    "recycles": conditions.get("recycles"),
                },
                "GPU rule": conditions.get("gpu_rule"),
                "MSA method": conditions.get("msa"),
                "MSA rows by chain": conditions.get("msa_rows_by_chain"),
                "paired MSA": conditions.get("paired_msa"),
                "quality metric": conditions.get("quality_metric"),
                "ColabFold settings": conditions.get("colabfold_settings"),
                "ESMFold 2 settings": conditions.get("esmfold2_settings"),
                "Chai-1 settings": conditions.get("chai1_settings"),
                "AF2-IG speed-only settings": conditions.get("af2ig"),
                "Excluded engines": conditions.get("excluded_engine_notes"),
            }
        )
        st.caption(f"Data folder: `{data['root']}`")
