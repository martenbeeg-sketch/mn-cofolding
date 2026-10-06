from __future__ import annotations

import streamlit as st

from mn_cofolding.inputs import build_af3_json
from mn_cofolding.foldbench_benchmark import render_foldbench_dashboard, render_foldbench_overview
from mn_cofolding.jobs import JobStore
from mn_cofolding.models import MODELS, get_model
from mn_cofolding.protenix_benchmark import (
    load_attempts as load_protenix_attempts,
    protenix_benchmark_table,
    render_protenix_attempt,
    results_root as protenix_results_root,
)
from mn_cofolding.references import all_reference_statuses
from mn_cofolding.result_page import boltz_benchmark_table, job_link_table, render_job_result
from mn_cofolding.runtime import get_settings
from mn_cofolding.worker import ensure_worker_service


st.set_page_config(page_title="MN Cofolding", page_icon="🧬", layout="wide")
settings = get_settings()
settings.ensure_dirs()
store = JobStore(settings)

job_page_id = st.query_params.get("job_id")
if job_page_id:
    render_job_result(str(job_page_id), store, settings)
    st.stop()

protenix_attempt_id = st.query_params.get("protenix_attempt")
if protenix_attempt_id:
    render_protenix_attempt(str(protenix_attempt_id), protenix_results_root())
    st.stop()

if st.query_params.get("foldbench_benchmark"):
    render_foldbench_dashboard()
    st.stop()

st.title("MN Cofolding")
st.caption("Submit protein and biomolecular folding jobs, then inspect or move their complete run folders.")

with st.form("cofold-submit"):
    left, right = st.columns([1, 1])
    with left:
        model_names = [model.name for model in MODELS]
        selected = st.selectbox(
            "Model",
            model_names,
            format_func=lambda value: next(model.label for model in MODELS if model.name == value),
        )
        selected_model = get_model(selected)
        job_name = st.text_input("Job name", placeholder="Optional label")
        msa_source_options = (
            ["local_mmseqs_gpu", "single_sequence"]
            if selected_model.name in {"esmfold2", "chai1"}
            else ["colabfold_server", "local_mmseqs_gpu", "single_sequence"]
        )
        msa_source_label = st.selectbox(
            "Protein MSA source",
            msa_source_options,
            format_func={
                "colabfold_server": "ColabFold MSA server",
                "local_mmseqs_gpu": "Local MMseqs-GPU (shared cache first)",
                "single_sequence": "Single sequence, no MSA",
            }.get,
        )
        msa_mode = "mmseqs2_uniref_env"
        if msa_source_label == "colabfold_server" and selected_model.family == "af2":
            msa_mode = st.selectbox(
                "Server MSA mode",
                ["mmseqs2_uniref_env", "mmseqs2_uniref", "mmseqs2_uniref_env_envpair"],
                format_func={
                    "mmseqs2_uniref_env": "UniRef + environmental",
                    "mmseqs2_uniref": "UniRef only",
                    "mmseqs2_uniref_env_envpair": "Paired environmental",
                }.get,
            )
        elif msa_source_label == "single_sequence":
            msa_mode = "single_sequence"
        if msa_source_label == "local_mmseqs_gpu":
            st.caption(
                f"Reuses A3Ms from `{settings.msa_cache_dir}`; missing MSAs are generated from "
                f"`{settings.reference_dir / 'alignment'}` with `mn-alphafast:cu128`. AF2 local-MSA jobs support one chain."
            )
        if selected_model.name == "esmfold2":
            st.caption(
                "Uses the pinned full Biohub ESMFold2 + ESM-C 6B weights from `reference_files`. "
                "The kit reads unpaired A3Ms per chain (maximum depth defaults to 1,024)."
            )
        if selected_model.name == "chai1":
            st.caption(
                "Uses native chai_lab 0.6.1 and the Chai-1 optimization kit. The kit reads local unpaired A3Ms per chain "
                "(maximum depth defaults to 1,024); `big` stays on one GPU."
            )
        input_options = (
            ["Molecule fields"]
            if selected_model.family == "af2" or selected_model.name in {"esmfold2", "chai1"}
            else ["Molecule fields", "Upload AlphaFold 3 JSON"]
        )
        input_kind = st.radio(
            "Input",
            input_options,
            horizontal=True,
        )
    with right:
        if input_kind == "Molecule fields":
            st.caption("Separate chains with colons. For example, `SEQ_A:SEQ_B`. A chain field can contain several chains.")
            protein = st.text_area("Protein sequences", placeholder="MKWVTFISLLFLFSSAYS …:MPEPTIDE …", height=90)
            dna = rna = ligand_ccd = ligand_smiles = ""
            if selected_model.name not in {"esmfold2", "chai1"} and selected_model.family != "af2":
                dna = st.text_input("DNA sequences", placeholder="ACGTACGT (optional)")
                rna = st.text_input("RNA sequences", placeholder="ACGUACGU (optional)")
                ligand_ccd = st.text_input("Ligand CCD codes", placeholder="ATP:HEM (optional)")
                ligand_smiles = st.text_input("Ligand SMILES", placeholder="CC(=O)O (optional)")
            uploaded_json = None
        else:
            uploaded_json = st.file_uploader("AlphaFold 3 dialect JSON", type=["json"])
            protein = dna = rna = ligand_ccd = ligand_smiles = ""
    with st.expander("Advanced run settings"):
        st.caption("Defaults match AlphaFold 3: 10 recycles and 5 samples. Increase them only if needed.")
        seeds_raw = st.text_input("Random seeds (comma-separated)", value="1")
        sample_label = "Models per seed" if selected_model.family == "af2" else "Diffusion samples per seed"
        num_diffusion_samples = st.number_input(sample_label, min_value=1, max_value=5, value=5, step=1)
        num_recycles = st.number_input(
            "Loop count" if selected_model.name == "esmfold2" else "Trunk recycle count" if selected_model.name == "chai1" else "Recycle count",
            min_value=1,
            max_value=100,
            value=3 if selected_model.name == "chai1" else 10,
        )
        msa_max_depth = 1024
        if selected_model.name in {"esmfold2", "chai1"}:
            msa_max_depth = st.number_input(
                "Maximum MSA rows per chain",
                min_value=1,
                max_value=8192,
                value=1024,
                step=128,
                help="Rows are capped when the native engine reads each A3M. The FoldBench comparison uses 1,024 rows per chain.",
            )
        optimization_mode = "off"
        if selected_model.name == "chai1":
            optimization_mode = st.selectbox(
                "Chai-1 optimization mode",
                ["off", "exact", "fast", "big"],
                help=(
                    "off is stock chai_lab; exact preserves stock output under the deterministic recipe; fast enables the kit's "
                    "faster math; big reduces GPU memory on one GPU."
                ),
            )
        elif selected_model.family == "af3":
            mode_options = (
                ["off", "exact", "fast", "big"]
                if selected_model.name in {"alphafold3", "esmfold2"}
                else ["off", "fast", "big"]
            )
            optimization_mode = st.selectbox(
                "ESMFold2 optimization mode" if selected_model.name == "esmfold2" else "AF3 optimization mode",
                mode_options,
                help=(
                    "ESMFold2 uses the kit's off/exact/fast/big modes; off, exact, and fast fold on GPU 0, while big row-shards one prediction across GPUs 0 and 1. "
                    "For ESMFold2, exact matches off with the kit's fused backend. fast may make small numerical changes; big is primarily a memory mode. "
                    if selected_model.name == "esmfold2"
                    else (
                        "off uses stock kernels. exact fuses the 128-channel pair GLU with the stock operation's rounding "
                        "(official AlphaFold 3 weights only); full predictions can differ from off. fast enables guarded "
                        "FlashPairformer kernels. big adds 256-row pair-transition shards and processes diffusion samples one at a time. "
                        "Official AlphaFold 3 big also row-shards its trunk across GPUs 0 and 1; other AF3-family big jobs stay on GPU 0. "
                        "The official AF3 path chunks pair conditioning and diffusion pair logits by rows; all AF3-family paths chunk triangle multiplication. "
                        "These memory steps may run slower and do not promise a speedup."
                    )
                ),
            )
        c1, c2 = st.columns(2)
        with c1:
            cpu_threads = st.number_input("CPU slots to reserve", min_value=1, max_value=256, value=4)
        with c2:
            use_gpu = st.checkbox("Reserve a GPU", value=True, disabled=selected_model.name in {"esmfold2", "chai1"})
            if selected_model.name in {"esmfold2", "chai1"}:
                use_gpu = True
    accept_terms = False
    if selected == "alphafold3":
        accept_terms = st.checkbox("I have reviewed and accept the official AlphaFold 3 weight terms of use")
        st.caption("The official AlphaFold 3 weight terms restrict use. The app will not start this model without explicit acceptance.")
    submitted = st.form_submit_button("Queue cofolding job", type="primary", width="stretch")

if submitted:
    try:
        seed_values = [int(part.strip()) for part in seeds_raw.split(",") if part.strip()]
        if input_kind == "Molecule fields":
            if selected_model.family == "af2":
                if any((dna, rna, ligand_ccd, ligand_smiles)):
                    raise ValueError("AlphaFold 2 accepts protein chains only")
                protein_chains = [part.strip() for part in protein.split(":") if part.strip()]
                if not protein_chains:
                    raise ValueError("Add at least one protein chain")
                input_text = f">{job_name or 'cofold'}\n{':'.join(protein_chains)}\n"
                input_format = "fasta"
            else:
                input_text = build_af3_json(
                    name=job_name or "cofold",
                    protein=protein,
                    dna=dna,
                    rna=rna,
                    ligand_ccd=ligand_ccd,
                    ligand_smiles=ligand_smiles,
                    seeds=seed_values,
                    msa_mode=msa_mode,
                )
                input_format = "alphafold3_json"
        else:
            if uploaded_json is None:
                raise ValueError("Choose an AlphaFold 3 JSON file first")
            input_text = uploaded_json.getvalue().decode("utf-8")
            input_format = "alphafold3_json"
        job = store.create(
            model_name=selected,
            input_text=input_text,
            input_format=input_format,
            name=job_name or None,
            msa_mode=msa_mode,
            msa_source=msa_source_label,
            seeds=seed_values,
            num_diffusion_samples=int(num_diffusion_samples),
            num_recycles=int(num_recycles),
            optimization_mode=optimization_mode,
            msa_max_depth=int(msa_max_depth),
            cpu_threads=int(cpu_threads),
            gpu=use_gpu,
            accept_alphafold3_terms=accept_terms,
        )
        ensure_worker_service(settings)
        st.success(f"Queued job `{job['metadata']['job_id']}` in `{job['run_dir']}`")
        st.markdown(f"[Open job results](?job_id={job['metadata']['job_id']})")
    except (ValueError, UnicodeDecodeError) as exc:
        st.error(str(exc))

st.divider()
all_jobs = store.list(limit=10000)
benchmark_jobs = [
    item for item in all_jobs
    if isinstance(item.get("benchmark"), dict)
    and item["benchmark"].get("experiment") == "boltz2-length-capacity-2026-10-01"
]
if benchmark_jobs:
    st.subheader("Boltz-2 long-protein benchmark")
    st.caption(
        "The main `off`/`fast`/`big` sweep uses the same cached A3M with Boltz capped at 8,192 rows on one RTX 4090. "
        "Large-target `big` runs also use two GPUs; reduced-MSA checks are labeled separately. "
        "Model time is shown for successful predictions; wall time includes failed attempts. "
        "Open a result to inspect the structure, confidence data, and logs."
    )
    st.markdown(boltz_benchmark_table(benchmark_jobs), unsafe_allow_html=True)

protenix_attempts = load_protenix_attempts()
if protenix_attempts:
    st.divider()
    st.subheader("Protenix optimization benchmark")
    st.caption(
        "Same cached MSA and Protenix v2 settings across modes. `off` and `fast` use GPU 0; "
        "`big` uses row-pair sharding across GPUs 0 and 1. Cells link to the attempt input, "
        "mode report, logs, timings, GPU memory, confidence data, and structure."
    )
    st.markdown(protenix_benchmark_table(protenix_attempts), unsafe_allow_html=True)
    st.caption(f"Structured attempt records and raw outputs: `{protenix_results_root()}`")

render_foldbench_overview()

st.divider()
st.subheader("Jobs")
jobs = store.list(limit=200)
if jobs:
    st.markdown(job_link_table(jobs), unsafe_allow_html=True)
    active_jobs = [item for item in jobs if item.get("status") in {"queued", "preparing", "running", "cancelling"}]
    if active_jobs:
        selected_job = st.selectbox(
            "Active job to cancel",
            [item["job_id"] for item in active_jobs],
            format_func=lambda value: next(item.get("name", value) for item in active_jobs if item["job_id"] == value),
        )
        if st.button("Cancel selected job"):
            try:
                store.cancel(selected_job)
                st.rerun()
            except ValueError as exc:
                st.warning(str(exc))
else:
    st.info("No folding jobs yet.")

with st.expander("Reference weights on this computer"):
    st.caption(f"Shared reference path: `{settings.reference_dir}`. Missing weights are fetched by the runner when selected.")
    st.caption(f"Shared MSA cache: `{settings.msa_cache_dir or settings.reference_dir / 'msa_cache'}`. A3Ms can be reused by MN Protein Design, MN Ligand, and MN Cofolding.")
    status_rows = all_reference_statuses(settings)
    st.dataframe(
        [{"Model": item.model, "Family": item.family, "Cache": item.state, "Path": item.path, "Details": item.detail} for item in status_rows],
        width="stretch",
        hide_index=True,
    )

st.caption(f"Runs are stored in `{settings.runs_dir}`. Their input, logs, metadata, and results can be exported with `mn-cofolding export`.")
