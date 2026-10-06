from __future__ import annotations

import subprocess
from pathlib import Path

from .jobs import _read_json
from .msa_cache import af2_msa_path, msa_cache_root
from .models import cli_model_type, get_model, image_for_model
from .runtime import Settings, get_settings


def docker_command(
    run_dir: Path,
    *,
    gpu_id: int | None = None,
    gpu_ids: list[int] | tuple[int, ...] | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Build the shell-free Docker command for one portable run folder."""
    settings = settings or get_settings()
    request = _read_json(run_dir / "input.json")
    model = get_model(str(request.get("model") or ""))
    image = image_for_model(
        model.name,
        af3_image=settings.af3_image,
        af2_image=settings.af2_image,
        esmfold2_image=settings.esmfold2_image,
        chai1_image=settings.chai1_image,
    )
    optimization_mode = str(request.get("optimization_mode", "off"))
    if (
        model.family == "af3"
        and optimization_mode != "off"
        and image == "mn-cofolding-af3:3.1.14-cu13"
    ):
        raise ValueError(
            "AF3 optimization modes require an AF3 optimization image (for example, mn-cofolding-af3-memory:3.1.14-cu13); "
            "the stock image supports off only"
        )
    if model.name == "alphafold3" and optimization_mode == "exact" and image != "mn-cofolding-af3-memory:3.1.14-cu13":
        raise ValueError(
            "AF3 exact mode requires mn-cofolding-af3-memory:3.1.14-cu13, "
            "the source-pinned official-weight image with the validated Pallas GLU kernel"
        )
    container = f"mn-cofolding-{str(request.get('job_id') or run_dir.name)}"
    command = [
        settings.docker_executable,
        "run",
        "--rm",
        "--name",
        container,
        "--shm-size=16g",
        "--mount",
        f"type=bind,src={run_dir.resolve()},dst=/work",
        "--mount",
        f"type=bind,src={settings.reference_dir.resolve()},dst=/reference",
        "-w",
        "/work",
    ]
    cpu_threads = str(request.get("cpu_threads", 4))
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        command.extend(["--env", f"{variable}={cpu_threads}"])
    selected_gpu_ids = list(gpu_ids or ([gpu_id] if gpu_id is not None else [0]))
    if request.get("gpu"):
        gpu_device_arg = ",".join(str(int(value)) for value in selected_gpu_ids)
        # The Docker CLI parses commas as request-field separators unless the
        # DeviceIDs value is quoted inside the flag argument. Keep the quote
        # characters in argv so `device=0,1` remains one value to Docker.
        command.extend(["--gpus", f'"device={gpu_device_arg}"'])

    if model.name == "esmfold2":
        if not request.get("gpu"):
            raise ValueError("ESMFold2 requires a reserved GPU")
        if optimization_mode == "big" and len(selected_gpu_ids) != 2:
            raise ValueError("ESMFold2 big mode requires GPU 0 and GPU 1")
        if optimization_mode != "big" and len(selected_gpu_ids) != 1:
            raise ValueError("ESMFold2 off/exact/fast modes use GPU 0 only")
        input_file = request.get("esmfold2_input_file")
        if not input_file:
            raise ValueError("ESMFold2 runner input has not been prepared")
        seeds = ",".join(str(int(seed)) for seed in (request.get("seeds") or [1]))
        kit_variant = str(request.get("esmfold2_variant") or "full_msa")
        if kit_variant not in {"full_msa", "full_nomsa"}:
            raise ValueError(f"Unsupported ESMFold2 variant: {kit_variant}")
        depth = int(request.get("msa_max_depth", 1024))
        config = esmfold2_config(selected_gpu_ids[0])
        command.extend(
            [
                "--mount",
                f"type=bind,src={settings.cache_dir.resolve()},dst=/cache",
                "--env",
                "HOME=/cache/home",
                "--env",
                "XDG_CACHE_HOME=/cache/xdg",
                "--env",
                "HF_HOME=/cache/huggingface/esmfold2",
                "--env",
                "MODEL_OPT_JIT_ROOT=/cache/esmfold2/jit",
                "--env",
                f"OMP_NUM_THREADS={cpu_threads}",
                "--env",
                "HF_HUB_ENABLE_HF_TRANSFER=0",
                image,
                "/usr/local/bin/mn-esmfold2-runner",
                "pred",
                "--config",
                config,
                "--variant",
                kit_variant,
                "--mode",
                optimization_mode,
                "--n_gpu",
                "2" if optimization_mode == "big" else "1",
                "--input",
                f"/work/{input_file}",
                "--out_dir",
                "/work/output",
                "--seeds",
                seeds,
                "--num_loops",
                str(request.get("num_recycles") or 10),
                "--num_sampling_steps",
                "200",
                "--num_diffusion_samples",
                str(request.get("num_diffusion_samples", 5)),
                "--msa_max_depth",
                str(depth),
                "--det",
                "1",
            ]
        )
        return command

    if model.name == "chai1":
        if not request.get("gpu"):
            raise ValueError("Chai-1 requires a reserved GPU")
        if len(selected_gpu_ids) != 1:
            raise ValueError("Native Chai-1 uses one GPU per fold in every mode")
        input_file = request.get("chai1_input_file")
        if not input_file:
            raise ValueError("Chai-1 runner input has not been prepared")
        depth = int(request.get("chai1_msa_depth") or request.get("msa_max_depth", 1024))
        config = chai1_config(selected_gpu_ids[0])
        command.extend(
            [
                "--mount",
                f"type=bind,src={settings.cache_dir.resolve()},dst=/cache",
                "--mount",
                f"type=bind,src={settings.reference_dir.resolve()},dst=/reference,readonly",
                "--env",
                "CHAI_DOWNLOADS_DIR=/reference/chai1",
                "--env",
                "MODEL_OPT_JIT_ROOT=/cache/chai1/jit",
                "--env",
                "TORCHINDUCTOR_CACHE_DIR=/cache/chai1/inductor",
                "--env",
                "CUDA_CACHE_PATH=/cache/chai1/cuda",
                "--env",
                f"OMP_NUM_THREADS={cpu_threads}",
                "--env",
                "OPENBLAS_NUM_THREADS=1",
                "--env",
                "MKL_NUM_THREADS=1",
                image,
                "/usr/local/bin/mn-chai1-runner",
                "pred",
                "--config",
                config,
                "--mode",
                optimization_mode,
                "--n_gpu",
                "1",
                "--input",
                f"/work/{input_file}",
                "--out_dir",
                "/work/output",
                "--tag",
                "chai1",
                "--seeds",
                ",".join(str(int(seed)) for seed in (request.get("seeds") or [1])),
                "--num-trunk-recycles",
                str(request.get("num_recycles") or 3),
                "--num-diffn-timesteps",
                "200",
                "--num-diffn-samples",
                str(request.get("num_diffusion_samples", 1)),
                "--num-trunk-samples",
                "1",
                "--recycle-msa-subsample",
                "0",
                "--no-use-msa-server",
                "--no-use-templates-server",
                "--low-memory",
                "--det",
                "1",
                "--msa-depth",
                str(depth),
            ]
        )
        return command

    if model.family == "af2":
        params_path = settings.reference_dir / "alphafold_models"
        (settings.cache_dir / "params").mkdir(parents=True, exist_ok=True)
        msa_key = str(request.get("msa_cache_key") or "")
        cached_msa = af2_msa_path(settings, str(request.get("msa_mode") or ""), msa_key) if msa_key else None
        local_msa_path = str(request.get("local_msa_path") or "")
        if local_msa_path:
            input_path = f"/msa_cache/{local_msa_path}"
        elif cached_msa and cached_msa.is_file():
            relative_cached = cached_msa.relative_to(msa_cache_root(settings))
            input_path = f"/msa_cache/{relative_cached.as_posix()}"
        else:
            input_path = f"/work/{request['input_file']}"
        command.extend(
            [
                "--mount",
                f"type=bind,src={settings.cache_dir.resolve()},dst=/cache",
                "--mount",
                f"type=bind,src={msa_cache_root(settings).resolve()},dst=/msa_cache",
                "--mount",
                f"type=bind,src={params_path.resolve()},dst=/cache/params",
                image,
                "colabfold_batch",
                input_path,
                "/work/output",
                "--data",
                "/cache",
                "--model-type",
                cli_model_type(model.name),
                "--msa-mode",
                str(request.get("msa_mode") or "mmseqs2_uniref_env"),
                "--num-models",
                str(request.get("num_diffusion_samples", 5)),
                "--random-seed",
                str((request.get("seeds") or [1])[0]),
                "--num-seeds",
                str(len(request.get("seeds") or [1])),
            ]
        )
        if request.get("num_recycles") is not None:
            command.extend(["--num-recycle", str(request["num_recycles"])])
        return command

    command.extend(
        [
            "--mount",
            f"type=bind,src={settings.cache_dir.resolve()},dst=/cache",
            "--mount",
            f"type=bind,src={msa_cache_root(settings).resolve()},dst=/msa_cache",
            "--env",
            "HOME=/cache/home",
            "--env",
            "XDG_CACHE_HOME=/cache/xdg",
            "--env",
            "HF_HOME=/cache/huggingface",
            "--env",
            "AF3_WEIGHTS_DIR=/cache/alphafold3/weights",
            "--env",
            "AF3_CCD_CACHE_DIR=/cache/alphafold3/ccd",
            "--env",
            "AF3_MSA_CACHE_DIR=/msa_cache/mn-cofolding/alphafold3",
            "--env",
            "AF3_SHARED_MSA_CACHE_DIR=/msa_cache",
            "--env",
            "MN_COFOLDING_AF3_OPT_MODE=" + optimization_mode,
        ]
    )
    if model.name == "alphafold3":
        selected_gpu_count = (
            len(selected_gpu_ids)
            if request.get("gpu") and optimization_mode == "big" and len(selected_gpu_ids) == 2
            else 1
        )
        command.extend(["--env", f"MN_COFOLDING_AF3_N_GPU={selected_gpu_count}"])
    attention, xla_flags = gpu_tuning(gpu_id)
    if xla_flags:
        command.extend(["--env", f"XLA_FLAGS={xla_flags}"])
    command.extend(
        [
            image,
            "python",
            "/usr/local/bin/cofolding-af3-runner.py",
            f"--json_path=/work/{request.get('engine_input_file') or request['input_file']}",
            f"--model={model.name}",
            "--norun_data_pipeline",
            "--output_dir=/work/output",
            "--force_output_dir",
            "--cache_dir=/cache/alphafold3",
            f"--num_recycles={request.get('num_recycles') or 10}",
            f"--num_diffusion_samples={request.get('num_diffusion_samples', 5)}",
            f"--flash_attention_implementation={attention}",
        ]
    )
    if model.name == "alphafold3":
        command.append("--model_dir=/reference/alphafold3")
    elif model.family == "af2":
        command.append("--model_dir=/reference/alphafold_models")
    if model.name in {"chai1", "esmfold2", "esmfold2_lm300m", "esmfold2_lm600m"}:
        command.append("--use_esm_embeddings")
    if request.get("msa_mode") != "single_sequence" and request.get("msa_source") != "local_mmseqs_gpu":
        command.append("--use_msa_server")
    if not request.get("gpu"):
        command.extend(["--jax_backend=cpu", "--nojit"])
    return command


def gpu_tuning(gpu_id: int | None, *, runner=subprocess.run) -> tuple[str, str]:
    """Use the preview notebook's attention/XLA settings for the selected GPU."""
    if gpu_id is None:
        return "xla", ""
    try:
        result = runner(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        capability = float(result.stdout.splitlines()[0].strip()) if result.returncode == 0 else 0.0
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        capability = 0.0
    if capability < 8.0:
        return "xla", "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
    if capability < 9.0:
        return "xla", "--xla_gpu_enable_triton_gemm=false"
    return "triton", "--xla_gpu_enable_triton_gemm=false"


def esmfold2_config(gpu_id: int | None, *, runner=subprocess.run) -> str:
    """Select the kit deployment config from the visible GPU's compute capability."""
    if gpu_id is None:
        raise ValueError("ESMFold2 needs a selected GPU")
    try:
        result = runner(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        capability = float(result.stdout.splitlines()[0].strip()) if result.returncode == 0 else 0.0
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        capability = 0.0
    if capability >= 12.0:
        return "rtx5090"
    if capability >= 8.9:
        return "rtx4090"
    raise ValueError(f"ESMFold2 image supports RTX 4090 (8.9) and RTX 5090 (12.0) GPUs; selected GPU reports {capability}")


def chai1_config(gpu_id: int | None, *, runner=subprocess.run) -> str:
    if gpu_id is None:
        return "rtx4090"
    try:
        result = runner(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(gpu_id)],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        name = result.stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return "rtx5090" if "5090" in name else "rtx4090" if "4090" in name else "h100"
