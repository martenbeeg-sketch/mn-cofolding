from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from mn_cofolding.result_page import (
    _combine_quality_plot_pngs,
    _quality_plot_artifact,
    _render_quality_plot_png,
)


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        ("pae", [[0.0, 15.0], [15.0, 30.0]]),
        ("plddt", [80.0, 90.0, 100.0]),
    ],
)
def test_quality_plot_fallback_is_a_png(kind: str, values: object) -> None:
    image = _render_quality_plot_png(kind, values)
    assert image.startswith(b"\x89PNG\r\n\x1a\n")


def test_quality_plots_use_the_same_figure_height_and_keep_their_aspects() -> None:
    plddt = _render_quality_plot_png("plddt", [80.0, 90.0, 100.0])
    pae = _render_quality_plot_png("pae", [[0.0, 15.0], [15.0, 30.0]])

    with Image.open(BytesIO(plddt)) as plddt_image, Image.open(BytesIO(pae)) as pae_image:
        assert plddt_image.height == pae_image.height
        assert plddt_image.width == pae_image.width * 2
        assert pae_image.width == pae_image.height


def test_quality_plot_artifact_finds_engine_generated_pngs(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pae = output_dir / "target_pae.png"
    plddt = output_dir / "target_plddt.png"
    pae.write_bytes(b"saved PAE image")
    plddt.write_bytes(b"saved pLDDT image")

    assert _quality_plot_artifact(tmp_path, "pae") == pae
    assert _quality_plot_artifact(tmp_path, "plddt") == plddt
    with pytest.raises(ValueError, match="Unknown quality plot type"):
        _quality_plot_artifact(tmp_path, "coverage")


def test_combined_quality_image_gives_both_plots_the_same_height() -> None:
    plddt = _render_quality_plot_png("plddt", [80.0, 90.0, 100.0])
    pae = _render_quality_plot_png("pae", [[0.0, 15.0], [15.0, 30.0]])
    combined = _combine_quality_plot_pngs([plddt, pae], height=300)

    with Image.open(BytesIO(combined)) as image:
        assert image.height == 300
        assert image.width > 900
