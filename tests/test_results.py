from __future__ import annotations

import json

import numpy as np
import pytest

from mn_cofolding.results import (
    format_duration,
    job_durations,
    parse_attempt_timings,
    prediction_records,
)
from mn_cofolding.result_page import boltz_benchmark_table, job_link_table


def test_af3_predictions_read_metrics_and_seed_timing(tmp_path):
    run_dir = tmp_path / "run"
    sample = run_dir / "output" / "prediction" / "seed-7_sample-0"
    sample.mkdir(parents=True)
    structure = sample / "prediction_seed-7_sample-0_model.cif"
    structure.write_text("data_prediction\n")
    # The AF3 runner also writes a root-level copy; expose the per-sample copy only.
    (run_dir / "output" / "prediction" / "prediction_model.cif").write_text("data_prediction\n")
    (sample / "prediction_seed-7_sample-0_summary_confidences.json").write_text(
        json.dumps({"ptm": 0.81, "iptm": 0.72, "ranking_score": 0.79})
    )
    (sample / "prediction_seed-7_sample-0_confidences.json").write_text(
        json.dumps({"atom_plddts": [90, 80], "pae": [[0, 4], [4, 0]], "max_pae": 4})
    )
    (run_dir / "stdout.log").write_text(
        "Running model inference with seed 7 took 12.50 seconds.\n"
        "Extracting 3 inference samples with seed 7 took 0.50 seconds.\n"
    )

    records = prediction_records(run_dir)
    timings = parse_attempt_timings(run_dir, "af3", samples_per_seed=3)

    assert len(records) == 1
    assert records[0]["label"] == "Seed 7, sample 1"
    assert records[0]["mean_plddt"] == pytest.approx(85, abs=0.05)
    assert records[0]["ptm"] == 0.81
    assert records[0]["pae"] == [[0, 4], [4, 0]]
    assert records[0]["contact_probs"] is None
    assert timings == [
        {
            "Attempt": "Seed 7",
            "Inference time": 12.5,
            "Output processing time": 0.5,
            "Predictions": 3,
            "Recycles": None,
        }
    ]


def test_esm_confidence_disabled_output_hides_zero_placeholders_and_keeps_contact_map(tmp_path):
    run_dir = tmp_path / "run"
    sample = run_dir / "output" / "prediction" / "seed-1_sample-0"
    sample.mkdir(parents=True)
    structure = sample / "prediction_seed-1_sample-0_model.cif"
    structure.write_text("data_prediction\n")
    (sample / "prediction_seed-1_sample-0_summary_confidences.json").write_text(
        json.dumps({"ptm": None, "iptm": None, "ranking_score": None, "fraction_disordered": 0.08, "has_clash": 0})
    )
    (sample / "prediction_seed-1_sample-0_confidences.json").write_text(
        json.dumps({"atom_plddts": [0, 0], "pae": [[None, None], [None, None]], "contact_probs": [[1, 0.2], [0.2, 1]]})
    )

    records = prediction_records(run_dir)

    assert len(records) == 1
    assert records[0]["mean_plddt"] is None
    assert records[0]["plddt_values"] == []
    assert records[0]["pae"] is None
    assert records[0]["contact_probs"] == [[1, 0.2], [0.2, 1]]
    assert records[0]["fraction_disordered"] == 0.08
    assert records[0]["has_clash"] == 0


def test_af2_predictions_match_score_files_and_parse_each_attempt(tmp_path):
    run_dir = tmp_path / "run"
    output = run_dir / "output"
    output.mkdir(parents=True)
    structure = output / "abc_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_003.pdb"
    structure.write_text("ATOM\n")
    (output / "abc_scores_rank_001_alphafold2_ptm_model_2_seed_003.json").write_text(
        json.dumps({"plddt": [70, 90], "pae": [[0, 2], [2, 0]], "ptm": 0.73})
    )
    (output / "log.txt").write_text(
        "alphafold2_ptm_model_2_seed_003 took 18.2s (4 recycles)\n"
    )

    records = prediction_records(run_dir)
    timings = parse_attempt_timings(run_dir, "af2")

    assert len(records) == 1
    assert records[0]["mean_plddt"] == 80
    assert records[0]["ptm"] == 0.73
    assert timings[0]["Attempt"] == "alphafold2_ptm_model_2_seed_003"
    assert timings[0]["Inference time"] == 18.2
    assert timings[0]["Predictions"] == 1
    assert timings[0]["Recycles"] == 4


def test_boltz2_predictions_read_npz_confidence_and_phase_timing(tmp_path):
    run_dir = tmp_path / "run"
    prediction_dir = run_dir / "output" / "boltz_results_TARGET" / "predictions" / "TARGET"
    prediction_dir.mkdir(parents=True)
    structure = prediction_dir / "TARGET_model_0.cif"
    structure.write_text("data_prediction\n")
    (prediction_dir / "confidence_TARGET_model_0.json").write_text(
        json.dumps({"confidence_score": 0.81, "ptm": 0.75, "iptm": 0.0, "complex_plddt": 0.8})
    )
    np.savez_compressed(prediction_dir / "plddt_TARGET_model_0.npz", plddt=np.asarray([0.7, 0.9]))
    np.savez_compressed(prediction_dir / "pae_TARGET_model_0.npz", pae=np.asarray([[0.0, 10.0], [10.0, 0.0]]))
    (run_dir / "stdout.log").write_text("PHASE item=TARGET lm_s=- trunk_s=8.5 sampler_s=2.0 conf_s=- total_s=11.25\n")

    records = prediction_records(run_dir)
    timings = parse_attempt_timings(run_dir, "af3", model_name="boltz2")

    assert len(records) == 1
    assert records[0]["mean_plddt"] == 80.0
    assert records[0]["plddt_values"] == [70.0, 90.0]
    assert records[0]["ptm"] == 0.75
    assert records[0]["ranking_score"] == 0.81
    assert records[0]["pae"] == [[0.0, 10.0], [10.0, 0.0]]
    assert records[0]["pae_plot_values"] == records[0]["pae"]
    assert records[0]["pae_downsampled"] is False
    assert timings[0]["Inference time"] == 11.25
    assert timings[0]["Attempt"] == "TARGET · Boltz2 inference"


def test_boltz2_oom_phase_is_not_reported_as_completed_inference(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(json.dumps({"success": False, "status": "failed"}))
    (run_dir / "stdout.log").write_text(
        "WARNING: ran out of memory, skipping batch\n"
        "PHASE item=LONG lm_s=- trunk_s=- sampler_s=- conf_s=- total_s=0.9\n"
    )

    assert parse_attempt_timings(run_dir, "af3", model_name="boltz2") == []


def test_boltz2_large_pae_is_downsampled_only_for_page_rendering(tmp_path):
    run_dir = tmp_path / "run"
    prediction_dir = run_dir / "output" / "boltz_results_LONG" / "predictions" / "LONG"
    prediction_dir.mkdir(parents=True)
    structure = prediction_dir / "LONG_model_0.cif"
    structure.write_text("data_prediction\n")
    (prediction_dir / "confidence_LONG_model_0.json").write_text(json.dumps({"confidence_score": 0.7}))
    matrix = np.full((1030, 1030), 8.0, dtype=np.float32)
    np.savez_compressed(prediction_dir / "pae_LONG_model_0.npz", pae=matrix)

    record = prediction_records(run_dir)[0]

    assert record["pae"] is None
    assert record["pae_downsampled"] is True
    assert record["pae_residue_count"] == 1030
    assert len(record["pae_plot_values"]) == 512
    assert all(len(row) == 512 for row in record["pae_plot_values"])
    assert record["max_pae"] == 8.0


def test_esmfold2_output_reads_kit_rows_pae_and_seed_timings(tmp_path):
    run_dir = tmp_path / "run"
    output = run_dir / "output"
    cif_dir = output / "cif_all"
    cif_dir.mkdir(parents=True)
    stem = "8JZT__full__s2_x3"
    structure = cif_dir / f"{stem}.cif"
    structure.write_text("data_prediction\n")
    np.savez_compressed(
        cif_dir / f"{stem}_pae.npz",
        pae=np.asarray([[0.0, 4.0], [4.0, 0.0]], dtype=np.float16),
        plddt=np.asarray([0.8, 0.9], dtype=np.float16),
        asym_id=np.asarray([1, 2]),
        mol_type=np.asarray([0, 0]),
    )
    rows = [
        {"pred_file": f"{stem}.cif", "seed": 2, "sample_index": 3, "iptm": 0.71, "ptm": 0.82, "wall_s": 14.5, "msa_depths": [1024, 700]}
    ]
    (output / "pred_rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (run_dir / "result.json").write_text(json.dumps({"success": True, "status": "completed"}))

    records = prediction_records(run_dir)
    timings = parse_attempt_timings(run_dir, "af3", model_name="esmfold2", samples_per_seed=5)

    assert len(records) == 1
    assert records[0]["label"] == "Seed 2, sample 4"
    assert records[0]["seed"] == "seed-2"
    assert records[0]["sample"] == 3
    assert records[0]["mean_plddt"] == pytest.approx(85, abs=0.05)
    assert records[0]["ptm"] == 0.82
    assert records[0]["iptm"] == 0.71
    assert records[0]["pae"] == [[0.0, 4.0], [4.0, 0.0]]
    assert timings[0]["Inference time"] == 14.5
    assert timings[0]["Attempt"] == "Seed 2"
    assert timings[0]["Predictions"] == 5


def test_job_durations_include_whole_job_queue_and_engine_time():
    durations = job_durations(
        {
            "created_at": "2026-09-30T10:00:00+00:00",
            "started_at": "2026-09-30T10:00:20+00:00",
            "status": "completed",
        },
        {"finished_at": "2026-09-30T10:01:05+00:00"},
    )

    assert durations == {"total": 65, "queue_and_setup": 20, "engine": 45}
    assert format_duration(65) == "1 min 5 s"


def test_job_list_renders_ids_as_result_links():
    table = job_link_table(
        [
            {
                "job_id": "job-123",
                "name": "test <job>",
                "model": "openfold3",
                "status": "completed",
            }
        ]
    )

    assert '<a href="?job_id=job-123">job-123</a>' in table
    assert "&lt;job&gt;" in table


def test_boltz_benchmark_table_shows_attempt_outcome_and_runtime_links():
    table = boltz_benchmark_table(
        [
            {
                "job_id": "boltz-ok",
                "benchmark": {
                    "experiment": "boltz2-length-capacity-2026-10-01",
                    "gene": "CHRNA7",
                    "sequence_length": 502,
                    "profile": "full_msa",
                    "profile_label": "Full cached MSA",
                    "mode": "off",
                    "gpu_count": 1,
                    "msa_rows_used": 8192,
                    "msa_rows_available": 13331,
                    "predicted": True,
                    "outcome": "predicted",
                    "inference_seconds": 17.5,
                    "wall_seconds": 51.0,
                },
            },
            {
                "job_id": "boltz-oom",
                "benchmark": {
                    "experiment": "boltz2-length-capacity-2026-10-01",
                    "gene": "LCT",
                    "sequence_length": 1927,
                    "profile": "full_msa",
                    "profile_label": "Full cached MSA",
                    "mode": "fast",
                    "gpu_count": 1,
                    "msa_rows_used": 8192,
                    "msa_rows_available": 24269,
                    "predicted": False,
                    "outcome": "OOM",
                    "inference_seconds": 0.7,
                    "wall_seconds": 25,
                },
            },
        ]
    )

    assert "Yes" in table
    assert "No · OOM" in table
    assert "No · OOM</td><td>—</td>" in table
    assert "17.5 s" in table
    assert "51.0 s" in table
    assert '?job_id=boltz-ok' in table
    assert '?job_id=boltz-oom' in table
