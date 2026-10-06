from __future__ import annotations

import csv
import html
import json
import math
import re
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import streamlit as st

from .models import get_model
from .portability import PortabilityError, export_job_archive
from .results import (
    format_duration,
    job_durations,
    output_files,
    parse_attempt_timings,
    prediction_records,
)
from .runtime import Settings


def _numeric_metric(value: object, *, percent: bool = False) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.1f}" if percent else f"{float(value):.3f}"


def _matrix_available(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(row, list)
            and len(row) == len(value)
            and all(isinstance(cell, (int, float)) and not isinstance(cell, bool) and math.isfinite(float(cell)) for cell in row)
            for row in value
        )
    )


def _quality_plot_artifact(run_dir: Path, kind: str) -> Path | None:
    """Find a quality PNG written by an engine such as ColabFold."""
    aliases = {
        "pae": ("pae", "aligned_error"),
        "plddt": ("plddt", "lddt"),
    }
    if kind not in aliases:
        raise ValueError(f"Unknown quality plot type: {kind}")
    output_dir = run_dir / "output"
    candidates = [
        path
        for path in output_dir.rglob("*.png")
        if any(alias in path.stem.lower() for alias in aliases[kind])
    ]
    return min(candidates, key=lambda path: (len(path.relative_to(output_dir).parts), path.name)) if candidates else None


def _render_quality_plot_png(
    kind: str,
    values: object,
    *,
    plddt_axis_label: str = "Atom",
    pae_downsampled: bool = False,
) -> bytes:
    """Render notebook-style quality plots with compact, consistent typography."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    figure = None
    buffer = BytesIO()
    try:
        with matplotlib.rc_context(
            {
                "font.size": 8,
                "axes.titlesize": 10,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
            }
        ):
            if kind == "pae":
                matrix = np.asarray(values, dtype=float)
                if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("PAE plot requires a square numeric matrix")
                figure, axis = plt.subplots(figsize=(6, 6))
                image = axis.imshow(matrix, cmap="bwr", vmin=0, vmax=30, interpolation="nearest")
                colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
                colorbar.set_label("PAE (Å)", fontsize=8)
                colorbar.ax.tick_params(labelsize=7)
                axis.set_xlabel("Scored residue (sampled)" if pae_downsampled else "Scored residue")
                axis.set_ylabel("Aligned residue (sampled)" if pae_downsampled else "Aligned residue")
                suffix = " · downsampled preview" if pae_downsampled else ""
                axis.set_title(f"Predicted Aligned Error (PAE){suffix}")
            elif kind == "plddt":
                plddt = np.asarray(values, dtype=float)
                if plddt.ndim != 1 or not plddt.size or not np.isfinite(plddt).all():
                    raise ValueError("pLDDT plot requires a non-empty finite vector")
                figure, axis = plt.subplots(figsize=(12, 6))
                axis.plot(np.arange(len(plddt)), plddt, lw=1.5, color="#1f77b4")
                axis.set_xlim(0, max(len(plddt) - 1, 1))
                axis.set_ylim(0, 100)
                for value in (50, 70, 90):
                    axis.axhline(value, ls="--", lw=0.7, color="grey", alpha=0.5)
                axis.set_xlabel(plddt_axis_label)
                axis.set_ylabel("pLDDT")
                axis.set_title(f"Predicted pLDDT per {plddt_axis_label.lower()}")
            else:
                raise ValueError(f"Unknown quality plot type: {kind}")
            figure.tight_layout()
            figure.savefig(buffer, format="png", dpi=120)
        return buffer.getvalue()
    finally:
        if figure is not None:
            plt.close(figure)


def _combine_quality_plot_pngs(images: list[bytes], *, height: int = 600, gap: int = 24) -> bytes:
    """Place plot PNGs in one row at equal height without distorting either plot."""
    if len(images) != 2:
        raise ValueError("Exactly two quality plots are needed for a paired image")

    from PIL import Image

    plots = [Image.open(BytesIO(data)).convert("RGB") for data in images]
    resized = [
        plot.resize(
            (max(1, round(plot.width * height / plot.height)), height),
            Image.Resampling.LANCZOS,
        )
        for plot in plots
    ]
    canvas = Image.new("RGB", (sum(plot.width for plot in resized) + gap, height), "white")
    x = 0
    for plot in resized:
        canvas.paste(plot, (x, 0))
        x += plot.width + gap
    result = BytesIO()
    canvas.save(result, format="PNG", optimize=True)
    return result.getvalue()


def _rank_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> tuple[Any, ...]:
        score = record.get("ranking_score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            return (0, -float(score), str(record["label"]))
        rank = re.search(r"\bRank (\d+)", str(record["label"]))
        if rank:
            return (1, int(rank.group(1)), str(record["label"]))
        if record.get("sample") is not None:
            return (2, int(record["sample"]), str(record.get("seed") or ""))
        return (3, str(record["label"]))

    return sorted(predictions, key=key)


def _elapsed_between(start: object, finish: object) -> float | None:
    if not start:
        return None
    try:
        begin = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    except ValueError:
        return None
    if begin.tzinfo is None:
        begin = begin.replace(tzinfo=timezone.utc)
    if finish:
        try:
            end = datetime.fromisoformat(str(finish).replace("Z", "+00:00"))
        except ValueError:
            end = None
        if end is not None and end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc)
    if end is None:
        return None
    return max(0.0, (end - begin).total_seconds())


def _archive_signature(run_dir: Path) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name != ".worker-claim.json":
            stat = path.stat()
            signature.append((path.relative_to(run_dir).as_posix(), stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


@st.cache_data(
    show_spinner="Preparing job ZIP…",
    ttl=3600,
    max_entries=8,
    hash_funcs={Settings: lambda value: str(value.runs_dir.resolve())},
)
def _cached_job_archive(job_id: str, settings: Settings, signature: tuple[tuple[str, int, int], ...]) -> bytes:
    del signature  # The signature invalidates this cached ZIP whenever a job artifact changes.
    return export_job_archive(job_id, settings)


@st.cache_data(
    show_spinner="Preparing interactive structure viewer…",
    ttl=3600,
    max_entries=12,
)
def _cached_py2dmol_html(
    run_dir: str,
    signature: tuple[tuple[str, int, int], ...],
    load_as_frames: bool,
    viewer_size: int = 520,
) -> str:
    """Build a self-contained py2Dmol viewer for all models in a job."""
    del signature  # The artifact signature invalidates the cached viewer when outputs change.
    from py2Dmol import view

    predictions = _rank_predictions(prediction_records(Path(run_dir)))
    has_maps = any(
        _matrix_available(record.get("pae")) or _matrix_available(record.get("contact_probs"))
        for record in predictions
    )
    has_plddt = any(record.get("plddt_values") for record in predictions)
    viewer = view(
        size=(viewer_size, viewer_size),
        controls=True,
        box=True,
        color="plddt" if has_plddt else "chain",
        style="richardson",
        bg="white",
        autoplay=load_as_frames,
        heatmap=has_maps,
        heatmap_size=viewer_size,
    )
    for index, record in enumerate(predictions, start=1):
        maps = {}
        if _matrix_available(record.get("contact_probs")):
            maps["contact"] = record["contact_probs"]
        viewer.add_pdb(
            str(record["structure_path"]),
            name="models" if load_as_frames else f"rank_{index}",
            paes=record.get("pae") if _matrix_available(record.get("pae")) else None,
            maps=maps or None,
        )
    return viewer.to_html(title="MN Cofolding prediction viewer")


def render_job_result(job_id: str, store, settings: Settings) -> None:
    try:
        detail = store.get(job_id)
    except (FileNotFoundError, ValueError):
        st.markdown("[← All jobs](./)")
        st.error(f"No job found with ID `{job_id}`.")
        return

    metadata = detail["metadata"]
    result = detail["result"]
    request = detail["input"]
    run_dir = detail["run_dir"]
    model = get_model(str(metadata.get("model") or request.get("model") or "openfold3"))
    status = str(metadata.get("status") or result.get("status") or "unknown")
    name = str(metadata.get("name") or request.get("name") or job_id)

    st.markdown("[← All jobs](./)")
    st.title(name)
    st.caption(f"Job ID: `{job_id}` · {model.label} · {status.replace('_', ' ').title()}")
    if st.button("Refresh job", key=f"refresh-{job_id}"):
        st.rerun()

    durations = job_durations(metadata, result)
    timings = parse_attempt_timings(
        run_dir,
        model.family,
        samples_per_seed=int(request.get("num_diffusion_samples", 5)),
        model_name=model.name,
    )
    inference_total = sum(
        float(row["Inference time"])
        for row in timings
        if isinstance(row.get("Inference time"), (int, float))
    )
    summary_cols = st.columns(4)
    summary_cols[0].metric("Status", status.replace("_", " ").title())
    summary_cols[1].metric("Whole job", format_duration(durations["total"]))
    engine_runtime = format_duration(durations["engine"]) if durations["engine"] is not None else "Not recorded"
    summary_cols[2].metric("Engine runtime", engine_runtime)
    summary_cols[3].metric("Inference attempts", format_duration(inference_total) if timings else "—")
    if status in {"queued", "preparing", "running", "cancelling"}:
        st.info("This job is still active. Use Refresh job to update its status and results.")
        if st.button("Cancel this job", key=f"cancel-{job_id}"):
            try:
                store.cancel(job_id)
                st.rerun()
            except ValueError as exc:
                st.warning(str(exc))
    elif status == "failed":
        st.error(str(result.get("message") or "The folding runner failed."))
    elif status == "cancelled":
        st.warning(str(result.get("message") or "This job was cancelled."))

    timing_details = []
    if durations["queue_and_setup"] is not None:
        timing_details.append(f"Queue and setup: {format_duration(durations['queue_and_setup'])}")
    msa_seconds = _elapsed_between(
        metadata.get("msa_preparation_started_at"),
        metadata.get("msa_preparation_finished_at"),
    )
    if metadata.get("msa_preparation_started_at"):
        timing_details.append(f"Local MSA preparation: {format_duration(msa_seconds)}")
    timing_details.extend(
        [
            f"Created: {metadata.get('created_at', 'unknown')}",
            f"Engine started: {metadata.get('started_at', 'not recorded for this job')}",
            f"Finished: {result.get('finished_at') or metadata.get('completed_at') or 'in progress'}",
        ]
    )
    st.caption(" · ".join(timing_details))

    predictions = _rank_predictions(prediction_records(run_dir))
    st.subheader("Display structures")
    if not predictions:
        st.info("No structure file has been written yet.")
    else:
        if len(predictions) > 1:
            load_as_frames = st.checkbox(
                "Load models as animation frames",
                value=False,
                key=f"viewer-frames-{job_id}",
                help="When off, choose a model from the viewer dropdown. When on, use the viewer's play, frame, speed, and overlay controls.",
            )
            if load_as_frames:
                st.caption("Use the play button and frame controls inside the viewer to animate the ranked predictions.")
        else:
            load_as_frames = False
            st.info(
                "Animation controls need at least two predictions. This job has one; request multiple diffusion samples "
                "or use multiple seeds in a new job to enable playback."
            )
        if any(_matrix_available(record.get("pae")) for record in predictions):
            st.caption(
                "The py2Dmol controls let you change the structure style and colours, rotate or focus the view, "
                "save an image, and click or drag on the PAE map to highlight residues. Both panels use square canvases."
            )
        elif any(_matrix_available(record.get("contact_probs")) for record in predictions):
            st.caption(
                "This model did not emit PAE. Its contact-probability map is shown instead; use the viewer controls "
                "to customize the structure, switch map tabs, and export an image. Both panels use square canvases."
            )
        else:
            st.caption(
                "Use the py2Dmol controls to customize the structure style and colours, orient or focus the view, "
                "and save an image. The structure canvas is square."
            )
        if any(record.get("pae_downsampled") for record in predictions):
            st.caption(
                "The full Boltz-2 PAE matrix is retained in the job files. For this long sequence, the quality plot "
                "uses a sampled preview to keep the interactive viewer responsive."
            )
        try:
            viewer_html = _cached_py2dmol_html(
                str(run_dir.resolve()),
                _archive_signature(run_dir),
                load_as_frames,
            )
            st.iframe(viewer_html, height=660, width="stretch")
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            st.error(f"Could not build the interactive prediction viewer: {exc}")

        with st.expander("Download an individual structure"):
            structure_labels = [record["label"] for record in predictions]
            structure_label = st.selectbox(
                "Prediction file",
                structure_labels,
                key=f"download-prediction-{job_id}",
            )
            structure = next(record["structure_path"] for record in predictions if record["label"] == structure_label)
            st.download_button(
                "Download selected structure",
                data=structure.read_bytes(),
                file_name=structure.name,
                mime="chemical/x-pdb" if structure.suffix.lower() == ".pdb" else "chemical/x-cif",
                key=f"download-structure-{job_id}-{structure_label}",
            )

    st.subheader("Quality metrics and plots")
    if not predictions:
        st.info("Quality metrics and plots will appear here when the engine writes prediction outputs.")
    else:
        rows = []
        for prediction in predictions:
            rows.append(
                {
                    "Prediction": prediction["label"],
                    "Mean pLDDT": _numeric_metric(prediction["mean_plddt"], percent=True),
                    "pTM": _numeric_metric(prediction["ptm"]),
                    "ipTM": _numeric_metric(prediction["iptm"]),
                    "Ranking score": _numeric_metric(prediction["ranking_score"]),
                    "Fraction disordered": _numeric_metric(prediction.get("fraction_disordered")),
                    "Clash detected": (
                        "—" if prediction.get("has_clash") is None
                        else "Yes" if prediction["has_clash"] >= 0.5 else "No"
                    ),
                    "Structure": prediction["relative_path"],
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)

        scored = any(
            prediction.get(key) is not None
            for prediction in predictions
            for key in ("mean_plddt", "ptm", "iptm", "ranking_score")
        )
        if scored:
            top_prediction = next(
                (
                    prediction for prediction in predictions
                    if any(prediction.get(key) is not None for key in ("mean_plddt", "ptm", "iptm", "ranking_score"))
                ),
                predictions[0],
            )
            st.markdown(f"**Top-ranked prediction:** {top_prediction['label']}")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Mean pLDDT", _numeric_metric(top_prediction["mean_plddt"], percent=True))
            metric_cols[1].metric("pTM", _numeric_metric(top_prediction["ptm"]))
            metric_cols[2].metric("ipTM", _numeric_metric(top_prediction["iptm"]))
            metric_cols[3].metric("Ranking score", _numeric_metric(top_prediction["ranking_score"]))
        elif model.name.startswith("esmfold2_lm"):
            st.info(
                "These ESMFold2 + ESM-C preview weights do not emit a confidence head. "
                "pLDDT, PAE, pTM, ipTM, and ranking scores are unavailable; zero pLDDT "
                "placeholders are hidden. The interactive viewer retains the contact-probability map."
            )
        else:
            st.info("This model did not write pLDDT, PAE, pTM, ipTM, or a ranking score for this prediction.")

        plddt_prediction = next((record for record in predictions if record.get("plddt_values")), None)
        pae_prediction = next(
            (
                record for record in predictions
                if _matrix_available(record.get("pae")) or _matrix_available(record.get("pae_plot_values"))
            ),
            None,
        )
        plddt_plot = _quality_plot_artifact(run_dir, "plddt")
        pae_plot = _quality_plot_artifact(run_dir, "pae")
        plot_images = []
        plot_sources = []
        if plddt_prediction:
            plot_images.append(
                _render_quality_plot_png(
                    "plddt",
                    plddt_prediction["plddt_values"],
                    plddt_axis_label="Residue" if model.name == "boltz2" else "Atom",
                )
            )
            plot_sources.append("pLDDT: rendered from saved confidence data")
        elif plddt_plot:
            plot_images.append(plddt_plot.read_bytes())
            plot_sources.append(f"pLDDT: {plddt_plot.relative_to(run_dir).as_posix()}")

        if pae_prediction:
            pae_values = pae_prediction.get("pae") or pae_prediction.get("pae_plot_values")
            plot_images.append(
                _render_quality_plot_png(
                    "pae",
                    pae_values,
                    pae_downsampled=bool(pae_prediction.get("pae_downsampled")),
                )
            )
            plot_sources.append(
                "PAE: sampled preview from saved confidence data"
                if pae_prediction.get("pae_downsampled")
                else "PAE: rendered from saved confidence data"
            )
        elif pae_plot:
            plot_images.append(pae_plot.read_bytes())
            plot_sources.append(f"PAE: {pae_plot.relative_to(run_dir).as_posix()}")

        plot_display_width: int | str = "stretch"
        if plot_images:
            fit_plots_to_page = st.checkbox(
                "Fit quality plots to page",
                value=True,
                key=f"fit-quality-plots-{job_id}",
                help="Turn this off to set an exact display width. Both plots keep their proportions and equal height.",
            )
            if not fit_plots_to_page:
                plot_display_width = int(
                    st.number_input(
                        "Quality plot width (pixels)",
                        min_value=800,
                        max_value=3000,
                        value=1800,
                        step=100,
                        key=f"quality-plot-width-{job_id}",
                    )
                )

        if len(plot_images) == 2:
            st.image(
                _combine_quality_plot_pngs(plot_images),
                caption=" · ".join(plot_sources),
                width=plot_display_width,
            )
        elif plot_images:
            st.image(plot_images[0], caption=" · ".join(plot_sources), width=plot_display_width)

        if pae_prediction:
            pae_note = " The preview samples 512 positions; the full matrix remains in the downloadable job files." if pae_prediction.get("pae_downsampled") else ""
            st.caption(
                "PAE uses the blue–white–red 0–30 Å scale. The PAE map and pLDDT chart are shown at the same height."
                + pae_note
            )
        elif any(_matrix_available(record.get("contact_probs")) for record in predictions):
            st.info("This engine did not emit PAE. Contact probabilities are shown in the interactive viewer instead.")

    st.subheader("Attempt timing")
    if timings:
        timing_rows = []
        for row in timings:
            timing_rows.append(
                {
                    "Attempt": row["Attempt"],
                    "Inference time": format_duration(row.get("Inference time")),
                    "Output processing": format_duration(row.get("Output processing time")),
                    "Predictions": row.get("Predictions", 1),
                    "Recycles": row.get("Recycles") if row.get("Recycles") is not None else "—",
                }
            )
        st.dataframe(timing_rows, width="stretch", hide_index=True)
        if model.name == "boltz2":
            st.caption("Boltz-2 reports completed model inference duration; whole-job wall time is shown above.")
        elif model.family == "af3":
            st.caption(
                "AlphaFold 3 reports one inference duration per seed. If a seed generated multiple diffusion samples, "
                "that duration covers the sample batch; output processing is shown separately."
            )
        else:
            st.caption("AlphaFold 2 reports each model/seed inference duration from the runner log.")
    elif model.name == "boltz2" and status == "failed":
        st.info("No prediction completed. Whole-job elapsed time above includes the failed attempt and cleanup.")
    else:
        st.info("Per-attempt timing was not recorded in this run's engine logs.")

    st.subheader("Run settings")
    gpu_ids = metadata.get("gpu_ids")
    if request.get("gpu") and isinstance(gpu_ids, list) and gpu_ids:
        gpu_setting = f"{', '.join(map(str, gpu_ids))} ({len(gpu_ids)} GPUs)"
    else:
        gpu_setting = str(metadata.get("gpu_id", "not assigned") if request.get("gpu") else "CPU")
    benchmark = metadata.get("benchmark") if isinstance(metadata.get("benchmark"), dict) else {}
    direct_kit_benchmark = benchmark.get("source") == "direct acceleration-kit image; result imported into MN Cofolding"
    settings_rows = [
        {"Setting": "Model", "Value": str(model.label)},
        {"Setting": "Status", "Value": str(status)},
        {"Setting": "Input", "Value": str(request.get("input_format", "unknown"))},
        {"Setting": "Protein chains", "Value": str(request.get("chain_count", "—"))},
        {"Setting": "Residues", "Value": str(request.get("residue_count", "—"))},
        {"Setting": "MSA source", "Value": str(request.get("msa_source", "colabfold_server"))},
        {"Setting": "MSA mode", "Value": str(request.get("msa_mode", "—"))},
        {"Setting": "Seeds", "Value": ", ".join(map(str, request.get("seeds") or [])) or "—"},
        {"Setting": "Samples per seed", "Value": str(request.get("num_diffusion_samples", 5))},
        {"Setting": "Recycle count", "Value": str(request.get("num_recycles") or "runner default")},
        {
            "Setting": "GPU",
            "Value": gpu_setting,
        },
    ]
    if direct_kit_benchmark:
        settings_rows.insert(-1, {"Setting": "Direct kit mode", "Value": str(benchmark.get("mode", "—"))})
        if benchmark.get("engine") == "af2ig":
            settings_rows.insert(-1, {"Setting": "Benchmark engine", "Value": "AF2-IG"})
        if benchmark.get("mode_levers_off"):
            settings_rows.insert(-1, {"Setting": "Kit levers withheld", "Value": str(benchmark["mode_levers_off"])})
    elif model.family == "af3":
        settings_rows.insert(
            -1,
            {
                "Setting": "Boltz-2 optimization mode" if model.name == "boltz2" else "AF3 optimization mode",
                "Value": str(request.get("optimization_mode", "off")),
            },
        )
    if model.name == "boltz2" and benchmark:
        rows_used = benchmark.get("msa_rows_used")
        rows_available = benchmark.get("msa_rows_available")
        msa_cap = benchmark.get("msa_cap")
        cap_text = f"{msa_cap:,}" if isinstance(msa_cap, int) else str(msa_cap or "—")
        msa_value = (
            f"{rows_used:,} / {rows_available:,} (cap {cap_text})"
            if isinstance(rows_used, int) and isinstance(rows_available, int)
            else "Not recorded"
        )
        settings_rows.extend(
            [
                {"Setting": "MSA rows used / available", "Value": msa_value},
                {"Setting": "Benchmark profile", "Value": str(benchmark.get("profile_label", "—"))},
                {"Setting": "Container image", "Value": str(benchmark.get("image", "—"))},
            ]
        )
    st.dataframe(settings_rows, width="stretch", hide_index=True)

    if benchmark.get("experiment") == "foldbench-heterodimer-length-2026-10-04":
        quality = benchmark.get("quality") if isinstance(benchmark.get("quality"), dict) else benchmark
        score_path = run_dir / "artifacts" / "quality" / "structure_scores.csv"
        threshold = quality.get(
            "score_threshold_DockQ",
            benchmark.get("quality_threshold_DockQ", 0.23),
        )
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = 0.23
        st.subheader("FoldBench DockQ-v2 quality")
        if benchmark.get("engine") == "af2ig":
            st.info(
                "Structure-informed refinement: the supplied native complex provides target context and the binder initial guess. "
                "The kit uses single-sequence AF2 features without an external MSA. This DockQ score records refinement from that input."
            )
        n_scored = int(quality.get("n_scored") or 0)
        if n_scored > 0:
            quality_cols = st.columns(4)
            quality_cols[0].metric("Mean DockQ", _numeric_metric(quality.get("mean_DockQ")))
            quality_cols[1].metric("Median DockQ", _numeric_metric(quality.get("median_DockQ")))
            quality_cols[2].metric("Best DockQ", _numeric_metric(quality.get("best_DockQ")))
            quality_cols[3].metric(
                "Successes",
                f"{quality.get('successes', 0)}/{quality.get('n_scored', 0)}",
            )
            st.caption(
                f"{benchmark.get('target', 'Target')} · {benchmark.get('mode', 'mode')} · "
                f"{quality.get('method', 'OpenStructure DockQ-v2 against the native two-chain assembly')} · "
                f"success cutoff DockQ ≥ {threshold:.2f}"
            )
        else:
            profile_status = quality.get("profile_status", benchmark.get("profile_status", "not available"))
            failure_reason = benchmark.get("failure_reason")
            message = f"This benchmark profile did not produce DockQ scores (profile status: {profile_status})."
            if failure_reason:
                message += f" {failure_reason}"
            st.info(message)
        if score_path.is_file():
            with score_path.open(newline="", encoding="utf-8") as score_file:
                score_rows = list(csv.DictReader(score_file))
            if score_rows:
                table_rows = []
                for row in score_rows:
                    table_rows.append(
                        {
                            "Seed": int(row["seed"]) if row.get("seed", "").isdigit() else row.get("seed", ""),
                            "Sample": int(row["sample"]) if row.get("sample", "").isdigit() else row.get("sample", ""),
                            "DockQ": float(row["DockQ"]) if row.get("DockQ") else None,
                            "iRMSD (Å)": float(row["iRMSD"]) if row.get("iRMSD") else None,
                            "LRMSD (Å)": float(row["LRMSD"]) if row.get("LRMSD") else None,
                            "Success ≥ 0.23": str(row.get("success", "")).lower() == "true",
                        }
                    )
                st.dataframe(
                    table_rows,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "DockQ": st.column_config.NumberColumn(format="%.3f"),
                        "iRMSD (Å)": st.column_config.NumberColumn(format="%.3f"),
                        "LRMSD (Å)": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
            with score_path.open("rb") as score_file:
                st.download_button(
                    "Download this profile's DockQ CSV",
                    data=score_file.read(),
                    file_name=f"{benchmark.get('target', job_id)}-{benchmark.get('mode', 'mode')}-DockQ.csv",
                    mime="text/csv",
                    key=f"foldbench-quality-{job_id}",
                )
        st.markdown("[Open FoldBench benchmark dashboard](?foldbench_benchmark=1)")

    if model.name == "boltz2":
        log_path = run_dir / "stdout.log"
        log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        optimization_lines = [
            line for line in log_text.splitlines()
            if "[mn-boltz2-opt" in line or "[boltz2-opt]" in line
        ]
        if benchmark.get("optimization_report"):
            optimization_lines.append(str(benchmark["optimization_report"]))
        with st.expander("Boltz-2 optimization details"):
            st.caption(
                f"Image: {benchmark.get('image', 'not recorded')} · profile: "
                f"{benchmark.get('profile_label', 'not recorded')}"
            )
            st.code("\n".join(optimization_lines) or "No runtime optimization report was recorded.", language="text")
    elif model.family == "af3":
        benchmark = metadata.get("benchmark") if isinstance(metadata.get("benchmark"), dict) else {}
        log_path = run_dir / "stdout.log"
        log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        optimization_lines = [
            line for line in log_text.splitlines()
            if "[mn-cofolding-af3-opt]" in line
        ]
        stored_report = benchmark.get("optimization_report")
        if stored_report and not any("report mode=" in line for line in optimization_lines):
            optimization_lines.append(str(stored_report))
        with st.expander("AF3 optimization details"):
            st.caption(
                f"Requested mode: {request.get('optimization_mode', 'off')}. "
                "The lines below record which kernels and memory hooks actually engaged."
            )
            st.code("\n".join(optimization_lines) or "No runtime optimization report was recorded.", language="text")

    active = status in {"queued", "preparing", "running", "cancelling"}
    download_col, files_col = st.columns([1, 2])
    with download_col:
        if active:
            st.button("Download job ZIP", disabled=True, help="The ZIP is available after the job finishes.")
        else:
            try:
                archive_bytes = _cached_job_archive(job_id, settings, _archive_signature(run_dir))
                st.download_button(
                    "Download job ZIP",
                    data=archive_bytes,
                    file_name=f"mn-cofolding-{quote(job_id, safe='')}.zip",
                    mime="application/zip",
                    key=f"download-job-{job_id}",
                    help="Includes the input, output structures, confidence data, logs, and job metadata.",
                )
            except (PortabilityError, OSError) as exc:
                st.error(f"Could not prepare the job ZIP: {exc}")
    with files_col:
        files = output_files(run_dir)
        st.caption(f"{len(files)} output file(s) · stored under `{run_dir}`")

    with st.expander("All output files"):
        for path in output_files(run_dir):
            st.code(path.relative_to(run_dir).as_posix(), language="text")
    with st.expander("Input, command, and logs"):
        st.markdown("**Input**")
        input_path = run_dir / str(request.get("input_file") or "")
        if input_path.is_file():
            st.code(input_path.read_text(errors="replace")[:30000], language="json" if input_path.suffix == ".json" else "text")
        command_path = run_dir / "command.json"
        if command_path.is_file():
            st.markdown("**Container command**")
            st.json(json.loads(command_path.read_text(encoding="utf-8")))
        for title, log_name in (("Runner stdout", "stdout.log"), ("Runner stderr", "stderr.log")):
            log_path = run_dir / log_name
            st.markdown(f"**{title}**")
            log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
            st.code(log_text[-30000:] or "No output", language="text")


def job_link_table(jobs: list[dict[str, Any]]) -> str:
    """Render a small HTML table whose job IDs link to their result views."""
    header = "<table><thead><tr><th>Job ID</th><th>Name</th><th>Model</th><th>Status</th><th>Created</th></tr></thead><tbody>"
    rows = []
    for item in jobs:
        job_id = str(item.get("job_id") or "")
        href = "?job_id=" + quote(job_id, safe="")
        rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(href, quote=True)}\">{html.escape(job_id)}</a></td>"
            f"<td>{html.escape(str(item.get('name') or ''))}</td>"
            f"<td>{html.escape(str(item.get('model') or ''))}</td>"
            f"<td>{html.escape(str(item.get('status') or ''))}</td>"
            f"<td>{html.escape(str(item.get('created_at') or ''))}</td>"
            "</tr>"
        )
    return header + "".join(rows) + "</tbody></table>"


def boltz_benchmark_table(jobs: list[dict[str, Any]]) -> str:
    """Render imported Boltz-2 capacity attempts with links to their full job pages."""
    mode_order = {"off": 0, "fast": 1, "big": 2}
    profile_order = {"full_msa": 0, "msa_cap_1024": 1, "msa_cap_512": 2}
    target_order = {
        "CHRNA7": 0,
        "SLC26A8": 1,
        "NUP155": 2,
        "LCT": 3,
        "EP300": 4,
        "CEP350": 5,
        "DMD": 6,
        "PRKDC": 7,
    }
    benchmark_jobs = [
        item for item in jobs
        if isinstance(item.get("benchmark"), dict)
        and item["benchmark"].get("experiment") == "boltz2-length-capacity-2026-10-01"
    ]
    benchmark_jobs.sort(
        key=lambda item: (
            target_order.get(str(item["benchmark"].get("gene")), 99),
            profile_order.get(str(item["benchmark"].get("profile")), 99),
            int(item["benchmark"].get("gpu_count", 1)),
            mode_order.get(str(item["benchmark"].get("mode")), 99),
        )
    )
    header = (
        "<table><thead><tr><th>Sequence</th><th>Length</th><th>Profile</th><th>Mode</th>"
        "<th>MSA rows used / available</th><th>Predicted</th><th>Model inference</th><th>Wall time</th><th>Result</th>"
        "</tr></thead><tbody>"
    )
    rows = []
    for item in benchmark_jobs:
        benchmark = item["benchmark"]
        job_id = str(item.get("job_id") or "")
        href = "?job_id=" + quote(job_id, safe="")
        gene = str(benchmark.get("gene") or "unknown")
        length = benchmark.get("sequence_length")
        mode = str(benchmark.get("mode") or "unknown")
        gpu_count = int(benchmark.get("gpu_count", 1) or 1)
        profile = str(benchmark.get("profile_label") or "full cached MSA")
        rows_used = benchmark.get("msa_rows_used")
        rows_available = benchmark.get("msa_rows_available")
        msa = f"{rows_used:,} / {rows_available:,}" if isinstance(rows_used, int) and isinstance(rows_available, int) else "—"
        predicted = bool(benchmark.get("predicted"))
        outcome = "Yes" if predicted else f"No · {benchmark.get('outcome', 'failed')}"
        model_time = format_duration(benchmark.get("inference_seconds")) if predicted else "—"
        wall_time = format_duration(benchmark.get("wall_seconds"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(gene)}</td>"
            f"<td>{html.escape(str(length if length is not None else '—'))} aa</td>"
            f"<td>{html.escape(profile)}</td>"
            f"<td>{html.escape(mode)} · {gpu_count} GPU{'s' if gpu_count != 1 else ''}</td>"
            f"<td>{html.escape(msa)}</td>"
            f"<td>{html.escape(outcome)}</td>"
            f"<td>{html.escape(model_time)}</td>"
            f"<td>{html.escape(wall_time)}</td>"
            f"<td><a href=\"{html.escape(href, quote=True)}\">Open result</a></td>"
            "</tr>"
        )
    return header + "".join(rows) + "</tbody></table>"
