from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    family: str
    description: str
    weights_note: str = ""


AF3_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("openbind0", "OpenBind 0", "af3", "OpenBind model weights in the shared AF3-family runner."),
    ModelSpec("openfold3", "OpenFold 3", "af3", "OpenFold 3 model weights in the shared AF3-family runner."),
    ModelSpec("boltz2", "Boltz-2", "af3", "Boltz-2 model weights in the shared AF3-family runner."),
    ModelSpec("protenix2", "Protenix 2", "af3", "Protenix 2 model weights in the shared AF3-family runner."),
    ModelSpec("rosettafold3", "RoseTTAFold 3", "af3", "RoseTTAFold 3 model weights in the shared AF3-family runner."),
    ModelSpec("chai1", "Chai-1", "af3", "Native chai_lab 0.6.1 with the Chai-1 optimization kit modes."),
    ModelSpec("intellifold2", "IntelliFold 2", "af3", "IntelliFold 2 model weights in the shared AF3-family runner."),
    ModelSpec("opendde", "OpenDDE", "af3", "OpenDDE model weights in the shared AF3-family runner."),
    ModelSpec("esmfold2", "ESMFold 2", "af3", "Biohub ESMFold2 full model with ESM-C 6B; optimized image supports all kit modes."),
    ModelSpec("esmfold2_lm600m", "ESMFold 2 + ESM-C 600M", "af3", "ESMFold 2 paired with its ESM-C 600M tower."),
    ModelSpec("esmfold2_lm300m", "ESMFold 2 + ESM-C 300M", "af3", "ESMFold 2 paired with its ESM-C 300M tower."),
    ModelSpec("alphafold3", "AlphaFold 3 (official weights)", "af3", "Official AlphaFold 3 parameters; use is subject to Google's weight terms."),
)

AF2_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("af2_ptm", "AlphaFold 2 pTM", "af2", "ColabFold AlphaFold 2 pTM model ensemble."),
    ModelSpec("af2_multimer", "AlphaFold 2 Multimer v3", "af2", "ColabFold AlphaFold 2 Multimer v3 model ensemble."),
)

MODELS = (*AF3_MODELS, *AF2_MODELS)
MODEL_BY_NAME = {model.name: model for model in MODELS}


def get_model(name: str) -> ModelSpec:
    try:
        return MODEL_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model {name!r}. Choose one of: {', '.join(MODEL_BY_NAME)}") from exc


def cli_model_type(model_name: str) -> str:
    get_model(model_name)
    return {
        "af2_ptm": "alphafold2_ptm",
        "af2_multimer": "alphafold2_multimer_v3",
    }.get(model_name, model_name)


def image_for_model(
    model_name: str,
    *,
    af3_image: str,
    af2_image: str,
    esmfold2_image: str = "mn-esmfold2-opt:0.1.0-cu130",
    chai1_image: str | None = None,
) -> str:
    if model_name == "esmfold2":
        return esmfold2_image
    if model_name == "chai1" and chai1_image is not None:
        return chai1_image
    return af3_image if get_model(model_name).family == "af3" else af2_image
