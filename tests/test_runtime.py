from __future__ import annotations

from mn_cofolding.runtime import Settings


def test_shared_msa_cache_defaults_under_reference_root(monkeypatch, tmp_path):
    references = tmp_path / "reference files"
    monkeypatch.setenv("MN_COFOLDING_REFERENCE_DIR", str(references))
    monkeypatch.delenv("MN_SHARED_MSA_CACHE_DIR", raising=False)
    monkeypatch.delenv("MN_COFOLDING_MSA_CACHE_DIR", raising=False)

    assert Settings.from_env().msa_cache_dir == references / "msa_cache"

    shared = tmp_path / "shared-msas"
    monkeypatch.setenv("MN_SHARED_MSA_CACHE_DIR", str(shared))
    assert Settings.from_env().msa_cache_dir == shared
