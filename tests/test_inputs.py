from __future__ import annotations

import json

import pytest

from mn_cofolding.inputs import build_af3_json, validate_input, validate_model_input_compatibility


EXAMPLE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"


def test_build_af3_json_preserves_notebook_molecule_order_and_fields():
    payload = json.loads(
        build_af3_json(
            name="mix test",
            protein=f"{EXAMPLE}:ACDE",
            rna="ACGU",
            dna="ACGT",
            ligand_ccd="ATP",
            ligand_smiles="C(=O)O",
            seeds=[3, 1, 3],
        )
    )
    assert payload["name"] == "mix_test"
    assert [next(iter(item)) for item in payload["sequences"]] == ["protein", "protein", "rna", "dna", "ligand", "ligand"]
    assert payload["modelSeeds"] == [3, 1]
    assert "unpairedMsa" not in payload["sequences"][0]["protein"]
    assert payload["sequences"][4]["ligand"]["ccdCodes"] == ["ATP"]
    assert payload["sequences"][5]["ligand"]["smiles"] == "C(=O)O"


def test_single_sequence_adds_inline_msa():
    payload = json.loads(build_af3_json(name="single", protein=EXAMPLE, msa_mode="single_sequence"))
    protein = payload["sequences"][0]["protein"]
    assert protein["unpairedMsa"] == f">query\n{EXAMPLE}\n"
    assert protein["pairedMsa"] == ""


@pytest.mark.parametrize("input_text", [">bad\nACD*\n", ">empty\n"])
def test_input_rejects_invalid_protein_fasta(input_text):
    with pytest.raises(ValueError):
        validate_input(input_text, "fasta", "openfold3")


def test_af2_rejects_json_input():
    with pytest.raises(ValueError, match="protein FASTA"):
        validate_model_input_compatibility("af2_ptm", "alphafold3_json")
