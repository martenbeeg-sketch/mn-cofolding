#!/usr/bin/env python3
"""Render mode-specific sequence-length matrices from the v2 image results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
ATTEMPTS = ROOT / "data" / "v2" / "all-attempts.csv"
PLOT_DIR = ROOT / "plots"
IMAGE_TAG = "mn-cofolding-af3-memory-v2:3.1.14-cu13"

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

MODEL_LABELS = {
    "alphafold3": "AlphaFold 3 · official weights",
    "openbind0": "OpenBind0",
    "openfold3": "OpenFold3",
    "boltz2": "Boltz-2",
    "protenix2": "Protenix2",
    "rosettafold3": "RoseTTAFold3",
    "chai1": "Chai-1",
    "intellifold2": "IntelliFold2",
    "opendde": "OpenDDE",
    "esmfold2": "ESMFold2",
    "esmfold2_lm300m": "ESMFold2 + ESM-C 300M",
    "esmfold2_lm600m": "ESMFold2 + ESM-C 600M",
}

# Values used in the image: 0 = not tested, 1 = all attempts failed,
# 2 = at least one completion with no failures, 3 = mixed outcomes.
COLORS = ["#e9edf1", "#dc7774", "#58ae82", "#e6b957"]
CMAP = ListedColormap(COLORS)


def load_attempts() -> list[dict[str, str]]:
    with ATTEMPTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("image") == IMAGE_TAG and row.get("mode") == "big"
    ]


def ordered_models(rows: list[dict[str, str]]) -> list[str]:
    best_completed: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("status") == "completed":
            best_completed[row["model"]] = max(
                best_completed[row["model"]], int(row["length_aa"])
            )
    return sorted(
        MODEL_LABELS,
        key=lambda model: (-best_completed[model], MODEL_LABELS[model].lower()),
    )


def draw_mode(mode: str, rows: list[dict[str, str]], model_order: list[str]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("mode") == mode:
            grouped[(row["model"], int(row["length_aa"]))].append(row)

    matrix = []
    annotations = []
    for model in model_order:
        values = []
        labels = []
        for _target, length in TARGETS:
            attempts = grouped.get((model, length), [])
            passed = sum(row.get("status") == "completed" for row in attempts)
            failed = sum(row.get("status") != "completed" for row in attempts)
            if not attempts:
                values.append(0)
                labels.append("—")
            elif passed and failed:
                values.append(3)
                labels.append(f"✓ / ×\n{passed}/{len(attempts)}")
            elif passed:
                values.append(2)
                labels.append(f"✓\n{passed}/{len(attempts)}")
            else:
                values.append(1)
                labels.append(f"×\n0/{len(attempts)}")
        matrix.append(values)
        annotations.append(labels)

    fig, ax = plt.subplots(figsize=(16, 8.4), dpi=180)
    ax.imshow(matrix, cmap=CMAP, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(
        [f"{gene}\n{length:,} aa" for gene, length in TARGETS],
        fontsize=9.5,
    )
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels([MODEL_LABELS[model] for model in model_order], fontsize=9.5)
    ax.set_xlabel("Tested single-chain protein sequence", fontsize=10, labelpad=10)
    ax.set_ylabel("Model / weights", fontsize=10, labelpad=10)
    ax.set_xticks([x - 0.5 for x in range(1, len(TARGETS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(model_order))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0, pad=8)

    for y, row in enumerate(annotations):
        for x, text in enumerate(row):
            ax.text(x, y, text, ha="center", va="center", fontsize=8.6, color="#17212b")

    mode_title = {"off": "OFF", "fast": "FAST", "big": "BIG"}[mode]

    if mode in {"off", "fast"}:
        pass_count = sum(row.get("mode") == mode for row in rows)
        subtitle = (
            f"RTX 4090 24 GB · {IMAGE_TAG} · {pass_count} {mode} attempts recorded; "
            "older-image results are archived separately."
        )
    else:
        profiles = sorted({row.get("environment_profile", "") for row in rows if row.get("mode") == mode})
        subtitle = (
            f"RTX 4090 24 GB · {IMAGE_TAG} · {sum(row.get('mode') == mode for row in rows)} attempts; "
            "one seed, 10 recycles, 5 samples."
        )
        if profiles:
            subtitle += " Profiles pooled: " + ", ".join(profiles) + "."

    fig.text(
        0.15,
        0.975,
        f"Long-protein execution capability · {mode_title} mode",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#17212b",
    )
    fig.text(0.15, 0.925, subtitle, ha="left", va="top", fontsize=8.7, color="#44515d")
    if mode in {"off", "fast"}:
        fig.text(
            0.15,
            0.89,
            "All cells are untested on v2; gray does not mean failure.",
            ha="left",
            va="top",
            fontsize=8.7,
            color="#44515d",
        )
    else:
        fig.text(
            0.15,
            0.89,
            "Cell labels show completed / attempted runs; mixed cells passed in some profiles and failed in others.",
            ha="left",
            va="top",
            fontsize=8.7,
            color="#44515d",
        )
    legend = [
        Patch(facecolor=COLORS[2], label="Completed in every recorded attempt"),
        Patch(facecolor=COLORS[3], label="Mixed: completed and failed attempts"),
        Patch(facecolor=COLORS[1], label="All recorded attempts failed"),
        Patch(facecolor=COLORS[0], label="Not tested on this image / mode"),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.15,
        0.015,
        "A completed run means prediction files were produced; it does not validate structural accuracy. "
        "These are tested sequence lengths, not guaranteed maximum-length limits.",
        ha="left",
        fontsize=8.5,
        color="#59636d",
    )
    fig.subplots_adjust(left=0.25, right=0.985, top=0.84, bottom=0.23)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / f"{mode}-capability-v2.png", dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = load_attempts()
    if not rows:
        raise SystemExit(f"No rows for image {IMAGE_TAG} in {ATTEMPTS}")
    models = ordered_models(rows)
    for mode in ("off", "fast", "big"):
        draw_mode(mode, rows, models)
    print(f"Rendered three PNG matrices to {PLOT_DIR}")


if __name__ == "__main__":
    main()
