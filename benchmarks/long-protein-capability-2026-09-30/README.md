# Long-protein capability matrices

These PNGs summarize the recorded capability sweep on one RTX 4090 24 GB.
The current-image plots use only `mn-cofolding-af3-memory-v2:3.1.14-cu13`.

## Plots

- `plots/off-capability-v2.png`
- `plots/fast-capability-v2.png`
- `plots/big-capability-v2.png`

The v2 archive contains 54 `big`-mode attempts across 12 model/weight choices.
It has no `off` or `fast` attempts, so those two images intentionally show
untested cells. Earlier off/fast/big measurements came from the distinct
`mn-cofolding-af3-memory:3.1.14-cu13` image. They are retained in
`data/historical/all-attempts-cross-image-audit.csv`, but are excluded from
these v2 matrices so image versions are not mixed in a comparison.

The big-mode plot combines recorded v2 profiles: 95% memory pool, 90% memory
pool, and command buffers disabled. A cell's fraction shows completed runs
over attempts for that model and sequence length. A mixed cell means the same
model/target combination passed in one profile and failed in another. The raw
attempt table preserves each profile, runtime, job ID, image tag, and error.

## Interpretation

Sequence lengths are discrete tested targets, not a continuous maximum-length
guarantee. “Completed” means output files were produced; it does not establish
structural accuracy. ESMFold2 variants do not use AF3-style recycle/diffusion
settings as native inputs, so their prediction path differs from the AF3-style
engines.

The v2 image is an experimental follow-up to the shared AF3 3.1.14 image. It
keeps the same base model code and weights while adding smaller memory chunks.
The current tests show that v2 can run more slowly; they do not measure improved
prediction quality. On the overlapping Protenix2/NUP155 case, v2 completed at a
95% memory pool, while both images failed at 90%, so that result does not
isolate the code change from the pool-size change.

## Data and regeneration

- `data/v2/all-attempts.csv`: the 54 v2-only attempts used for the matrix.
- `data/v2/capability-summary.csv`: v2 profile-specific capability summary.
- `data/v2/manifests/`: v2 run manifests referenced by the attempt table.
- `data/v2/sweep-report.md`: the full v2 benchmark report.
- `data/historical/all-attempts-cross-image-audit.csv`: image-tagged archive
  covering both the older memory image and v2.
- `data/historical/manifests/`: manifests for earlier off/fast/big comparisons
  on the older `mn-cofolding-af3-memory` image.
- `plot_capabilities.py`: regenerates the three PNG files with Matplotlib.

Run from this directory with Python 3 and Matplotlib installed:

```bash
python3 plot_capabilities.py
```

The actual per-job structures, logs, and MSA files remain in the configured
results work directory. Their job IDs are recorded in the CSV for lookup.
