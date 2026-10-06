# MN Cofolding

MN Cofolding is a standalone Streamlit app for folding and biomolecular cofolding. It keeps each request, log, command record, and result under one job ID, exposes the same job store through a small REST API and CLI, and uses the shared `mn-compute-scheduler` CPU/GPU leases.

Each job ID in the app is a link to its result page. The page shows interactive structures, available confidence metrics and residue maps, per-seed/model timing, whole-job timing, run settings, logs, and a ZIP download containing that job's inputs and results. The embedded py2Dmol controls can change the structure style and colours, switch models or play them as frames, select residues on PAE/contact maps, and save images; the structure and map canvases are square.

The AF3-family engines use the preview fork's single `run_alphafold.py` entry point. AlphaFold 2 pTM and Multimer use the existing `mn-colabfold` image. The AF3 image, including the preview fork compatibility changes, is built from the central recipes in `mn-tool-containers`.

## Install

On a Linux machine with Conda or Mamba, Docker, and the NVIDIA Container Toolkit:

```bash
conda env create -f environment.yml
conda activate mn-cofolding
mn-cofolding init
```

Build the stock image, the preserved first optimized image, and the versioned memory image from the sibling `mn-tool-containers` checkout:

```bash
../mn-tool-containers/build.sh mn-cofolding
```

The build creates stock `mn-cofolding-af3:3.1.14-cu13`, the earlier
`mn-cofolding-af3-opt:3.1.14-cu13`, the tested
`mn-cofolding-af3-memory:3.1.14-cu13`, and the experimental
`mn-cofolding-af3-memory-v2:3.1.14-cu13`. MN Cofolding defaults to the tested
memory image. Set `MN_COFOLDING_AF3_IMAGE` to select v2 or another local tag.
All variants share their unchanged base layers on the same computer.

Choose `off`, `fast`, or `big` in the job's advanced settings. Official
AlphaFold 3 also offers `exact`, which fuses the 128-channel pair GLU with a
guarded Pallas kernel that preserves the stock GLU operation's arithmetic.
That exactness claim is scoped to the GLU operator: the 8OSW end-to-end CIFs
from `exact` differ from the matched `off` run. All four use a common 64-token
bucket policy so the modes run the same padded shape. `off` uses stock model
kernels. `fast` enables guarded fused triangle multiplication and triangle
attention. In the default memory image, `big` shards pair transitions into
256-row blocks and processes diffusion samples one at a time. Official AF3
`big` also row-shards the trunk across GPUs 0 and 1; other AF3-family models
stay on GPU 0. The model's 3.1.14 diffusion, confidence, and output paths stay
stock. Only `big` can reserve GPU 1, and the app records both device IDs.
The exact path is restricted to official AlphaFold 3 weights because its
source guard and operator-level check have only been validated there.
The experimental v2 image adds 64-row triangle multiplication, pair
conditioning, and diffusion-logit chunks. Both keep the original model weights
and diffusion schedule; `big` can be slower than `fast`.
AF2 remains on its separate ColabFold image and does not expose these modes.
Set `MN_COFOLDING_AF3_IMAGE` to use a different local tag; the stock tag
supports `off` only.

Fast mode's guarded tile settings have been exercised on the RTX 4090
(compute capability 8.9). Some 256-channel operations fall back to stock
kernels on that card because their fused kernels exceed its shared-memory
limit. With the default memory image and official AF3 weights, LCT (1927 aa)
and EP300 (2414 aa) completed in `big` at the default 10 recycles and 5 samples,
in 5.3 and 8.7 minutes. CEP350 (3117 aa) failed in `big` at 1/1. The v2 image
also failed CEP350 at 1/1 with both 95% and 99% JAX memory pools; AF3 requested
a 23.54 GiB allocation, and its memory planner estimated a peak above this
24 GB card. V2 completed LCT and EP300, but slower, in 7.3 and 11.4 minutes, so
it is retained as an optional experiment and is not the default. DMD (3685 aa)
and PRKDC (4128 aa) also failed at 1/1 in the default memory image. EP300
failed in `off` and `fast` even at 1/1 on the earlier optimized image. On this
RTX 4090, EP300 is the longest tested target that completed with official
weights. Result pages show the image tag, runtime optimization report, and
measured timings.

The dated [long-protein benchmark archive](benchmarks/long-protein-capability-2026-09-30/README.md)
keeps the v2 attempt table, image-tagged historical results, run manifests, and
Matplotlib capability matrices. The v2 sweep measured `big`; its `off` and
`fast` matrices show untested cells rather than mixing in older-image runs.

Start the UI:

```bash
mn-cofolding app --host 0.0.0.0
```

The REST API is available with `mn-cofolding api --host 0.0.0.0`; use `mn-cofolding --help` for job submission, worker, status, export, and import commands. Keep the Conda environment and Docker images on the host; containers run each model job.

## Persistent files and portable runs

By default, jobs and results live under `/mnt/data/RESULTS/mn-cofolding-workdir/workdir/runs/cofolding/<job-id>`. Shared model files are read from `/mnt/db/reference_files`. The existing `alphafold3/af3.bin.zst` and AlphaFold 2 pTM/Multimer parameter files are reused from there.

The shared MSA repository defaults to `/mnt/db/reference_files/msa_cache`, directly under the reference root. MN Protein Design, MN Ligand, MN Benchmark history, and MN Cofolding use this location; legacy paths such as `boltz_models/msa_repository` remain aliases for existing jobs. Existing A3Ms are retained, and new single-protein A3Ms use a sequence-hashed filename so other apps can find them. Set `MN_SHARED_MSA_CACHE_DIR` to point every app at a different shared location.

The model/runtime cache defaults to `/mnt/db/reference_files/mn-cofolding/cache`. It holds preview-format weights and ESM towers, JAX compilation artifacts, and the AF3 CCD tables. Set `MN_COFOLDING_CACHE_DIR` to move it. Keep or copy both cache roots when transferring installations between computers.

To move run records and structures, stop active jobs and use:

```bash
mn-cofolding export /path/to/cofolding-runs.zip
mn-cofolding verify /path/to/cofolding-runs.zip
mn-cofolding import /path/to/cofolding-runs.zip
```

The archive contains job inputs, metadata, logs, and output files, with checksums. It does not contain the container images, reference weights, or cache; those can be moved separately or downloaded on the target computer. Configure another work disk with `MN_COFOLDING_APP_HOME` and `MN_COFOLDING_RUN_DIR`, and configure shared weights/cache with `MN_COFOLDING_REFERENCE_DIR` and `MN_COFOLDING_CACHE_DIR`.

## Models and weights

The preview model options are `openbind0`, `openfold3`, `boltz2`, `protenix2`, `rosettafold3`, `chai1`, `intellifold2`, `opendde`, `esmfold2`, `esmfold2_lm600m`, `esmfold2_lm300m`, and `alphafold3`, plus `af2_ptm` and `af2_multimer`. The preview weights and language-model towers are populated lazily in the shared cache. Confidence metrics depend on the selected model weights: in the current preview, the base `esmfold2` variant writes PAE and scores, while the `esmfold2_lm300m` and `esmfold2_lm600m` variants do not emit a confidence head. For those two variants the result page marks pLDDT/PAE/pTM/ipTM/ranking scores unavailable and displays their contact-probability map instead. Official AlphaFold 3 weights are already present under `/mnt/db/reference_files/alphafold3`; the app asks for explicit user acknowledgement of their published terms before queuing that model.

`chai1` runs native `chai_lab 0.6.1` from the Chai-1 optimization-kit image (`mn-chai1-opt:0.6.1-cu130`) with `off`, `exact`, `fast`, and `big` modes. Its pinned weights are kept outside the image at `/mnt/db/reference_files/chai1`; native Chai reads local, unpaired per-chain A3Ms converted to its aligned parquet format. All modes, including `big`, use one GPU per fold.

When the Protenix optimization benchmark records are available under
`/mnt/data/RESULTS/protenix-optimization-playbook-20261001`, the app shows an
`off`/`fast`/`big` capacity matrix with a result page for each recorded attempt.
Set `MN_PROTENIX_BENCHMARK_DIR` if the records live elsewhere. Those pages expose
the exact input, image and mode, logs, timings, GPU samples, confidence data,
and predicted CIF files.

The FoldBench heterodimer benchmark is shown on the MN Cofolding home page and
opens as an interactive dashboard with target-length speed and DockQ plots,
per-GPU memory, profile filters, score downloads, and links to the registered
job pages. The default records are read from
`/mnt/data/RESULTS/mn-cofolding-workdir/workdir/benchmarks/foldbench-heterodimer-length-20261004`;
set `MN_FOLDBENCH_BENCHMARK_DIR` to use another folder containing
`benchmark_results.json` and `registered_jobs.json`. Each registered FoldBench
job page also displays its per-structure DockQ-v2 scores and a CSV download.

The one-page UI accepts protein, DNA, RNA, CCD ligand, and SMILES inputs. Protein chains are separated with colons. For protein MSAs, select the ColabFold server, local MMseqs-GPU, or single-sequence mode. Local MMseqs-GPU checks the shared repository first, then generates missing A3Ms with the installed `mn-alphafast:cu128` image and databases under `/mnt/db/reference_files/alignment/mmseqs`. It supports AF3-family complexes; the AF2 local input path currently supports one protein chain per job. Build the optional image with `MN_BUILD_LOCAL_MSA=1 ../mn-tool-containers/build.sh mn-cofolding`.

## Tests and model smoke run

Run the unit/API suite in the app environment:

```bash
pytest
```

After building both images, `python scripts/smoke_all_models.py` queues the preview notebook's example sequence (`PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK`) once with each model, one seed, one diffusion sample, and one recycle, then checks for a structure file. It runs on the available GPU by default and populates the cache on first use. Add `--msa-source local_mmseqs_gpu` to use the local MMseqs-GPU database instead of the ColabFold server. Pass `--accept-alphafold3-terms` only after reviewing and accepting the official AlphaFold 3 weight terms to include the official model in that smoke run.

The AF3 image uses JAX's CUDA 13 runtime and includes an optional Blackwell presence check. This machine's GPU smoke run is on an RTX 4090; run the following on an RTX 5090 to check that hardware directly:

```bash
docker run --rm --gpus all -e MN_REQUIRE_BLACKWELL=1 mn-cofolding-af3:3.1.14-cu13 python /usr/local/bin/cofolding-gpu-smoke-test
```

This checks that JAX sees a Blackwell GPU and can execute a GPU operation. It does not download or run model weights.
