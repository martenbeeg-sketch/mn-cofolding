from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProteinChain:
    id: str
    sequence: str
    ptms: tuple = ()
    description: str = ""
    unpaired_msa: str | None = None
    paired_msa: str | None = None
    templates: list | None = None


@dataclass
class RnaChain:
    id: str
    sequence: str
    modifications: list = field(default_factory=list)
    description: str = ""
    unpaired_msa: str | None = None


@dataclass
class FoldInput:
    chains: list

    @property
    def protein_chains(self):
        return [chain for chain in self.chains if isinstance(chain, ProteinChain)]

    @property
    def rna_chains(self):
        return [chain for chain in self.chains if isinstance(chain, RnaChain)]


def test_af3_unpaired_and_paired_msas_are_reused_across_runs(tmp_path, monkeypatch):
    wrapper_path = (
        Path(__file__).resolve().parents[2]
        / "mn-tool-containers"
        / "protein-design"
        / "cofolding-af3"
        / "cofolding-af3-runner.py"
    )
    spec = importlib.util.spec_from_file_location("cofolding_af3_runner_test", wrapper_path)
    wrapper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wrapper)

    calls = []
    msa_server = types.ModuleType("alphafold3.data.msa_server")

    def query(sequences, *, use_pairing=False, host_url="", user_agent=""):
        calls.append((tuple(sequences), use_pairing, host_url, user_agent))
        prefix = "paired" if use_pairing else "unpaired"
        return [f">{prefix}\n{sequence}\n" for sequence in sequences]

    msa_server._query_server = query
    msa_server.fill_missing_msas = lambda *_args, **_kwargs: None
    folding_input = types.ModuleType("alphafold3.common.folding_input")
    folding_input.ProteinChain = ProteinChain
    folding_input.RnaChain = RnaChain
    common = types.ModuleType("alphafold3.common")
    common.folding_input = folding_input
    data = types.ModuleType("alphafold3.data")
    data.msa_server = msa_server
    root = types.ModuleType("alphafold3")
    root.__path__ = []
    monkeypatch.setitem(sys.modules, "alphafold3", root)
    monkeypatch.setitem(sys.modules, "alphafold3.common", common)
    monkeypatch.setitem(sys.modules, "alphafold3.common.folding_input", folding_input)
    monkeypatch.setitem(sys.modules, "alphafold3.data", data)
    monkeypatch.setitem(sys.modules, "alphafold3.data.msa_server", msa_server)
    monkeypatch.setenv("AF3_MSA_CACHE_DIR", str(tmp_path / "msa" / "mn-cofolding" / "alphafold3"))
    monkeypatch.setenv("AF3_SHARED_MSA_CACHE_DIR", str(tmp_path / "msa"))

    wrapper._install_msa_cache()
    original = FoldInput(
        [ProteinChain(id="A", sequence="AAAA"), ProteinChain(id="B", sequence="BBBB")]
    )
    first = msa_server.fill_missing_msas(original)
    second = msa_server.fill_missing_msas(original)

    assert len(calls) == 2
    assert calls[0][1] is False
    assert calls[1][1] is True
    assert [chain.unpaired_msa.splitlines()[0] for chain in first.protein_chains] == [">unpaired", ">unpaired"]
    assert [chain.paired_msa.splitlines()[0] for chain in second.protein_chains] == [">paired", ">paired"]


def test_af3_reads_msa_created_by_another_app(tmp_path, monkeypatch):
    wrapper_path = (
        Path(__file__).resolve().parents[2]
        / "mn-tool-containers"
        / "protein-design"
        / "cofolding-af3"
        / "cofolding-af3-runner.py"
    )
    spec = importlib.util.spec_from_file_location("cofolding_af3_shared_test", wrapper_path)
    wrapper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wrapper)

    calls = []
    msa_server = types.ModuleType("alphafold3.data.msa_server")
    msa_server._query_server = lambda *args, **kwargs: calls.append((args, kwargs))
    msa_server.fill_missing_msas = lambda *_args, **_kwargs: None
    folding_input = types.ModuleType("alphafold3.common.folding_input")
    folding_input.ProteinChain = ProteinChain
    folding_input.RnaChain = RnaChain
    common = types.ModuleType("alphafold3.common")
    common.folding_input = folding_input
    data = types.ModuleType("alphafold3.data")
    data.msa_server = msa_server
    root = types.ModuleType("alphafold3")
    root.__path__ = []
    monkeypatch.setitem(sys.modules, "alphafold3", root)
    monkeypatch.setitem(sys.modules, "alphafold3.common", common)
    monkeypatch.setitem(sys.modules, "alphafold3.common.folding_input", folding_input)
    monkeypatch.setitem(sys.modules, "alphafold3.data", data)
    monkeypatch.setitem(sys.modules, "alphafold3.data.msa_server", msa_server)
    shared_root = tmp_path / "msa"
    shared_root.mkdir()
    import hashlib

    sequence = "AAAA"
    shared_path = shared_root / f"{hashlib.sha256(sequence.encode()).hexdigest()}.a3m"
    shared_path.write_text(">query\nAAAA\n>hit\nAAAA\n")
    monkeypatch.setenv("AF3_MSA_CACHE_DIR", str(tmp_path / "msa" / "mn-cofolding" / "alphafold3"))
    monkeypatch.setenv("AF3_SHARED_MSA_CACHE_DIR", str(shared_root))

    wrapper._install_msa_cache()
    result = msa_server.fill_missing_msas(FoldInput([ProteinChain(id="A", sequence=sequence)]))

    assert not calls
    assert result.protein_chains[0].unpaired_msa.startswith(">query\nAAAA")
