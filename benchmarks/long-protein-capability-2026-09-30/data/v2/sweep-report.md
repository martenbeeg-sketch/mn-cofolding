# Unified AF3 image capability sweep

## Scope

This report evaluates the unified JAX AF3 runner with the v2 image only: `mn-cofolding-af3-memory-v2:3.1.14-cu13`. The main CSVs contain only jobs from that image. It includes the image-default big-mode run (95% JAX GPU memory pool) and explicitly labeled tuning checks at 90% and with XLA command buffers disabled.

The run set contains 54 v2 attempts: 25 at the default 95% pool, 24 at a 90% override, and 5 with command buffers disabled. Nineteen completed and 35 failed. They ran on one RTX 4090 24GB GPU at a time, GPU 0. The submitted job settings were one seed, 10 recycles, 5 diffusion samples, and the ColabFold MSA server. Engines that do not use a setting follow their native inference path; ESMFold2 is a single-sequence language model and does not use AF3-style MSA, recycle, or diffusion inputs. The targets were single-chain sequences: CHRNA7 (502 aa), SLC26A8 (970 aa), NUP155 (1,391 aa), LCT (1,927 aa), EP300 (2,414 aa), CEP350 (3,117 aa), DMD (3,685 aa), and PRKDC (4,128 aa).

A completed job means the runner wrote a prediction successfully. It does not establish structural accuracy. ESMFold2 and its LM variants do not provide the same pTM/PAE/ranking outputs as the AF3-style models; those missing scores are an engine output limitation.

## v2 default: big mode, 95% memory pool

“Next failure” is the next tested larger sequence that failed in this profile. A dash means no larger target was tested for that engine in this profile.

| Engine / weights | Largest successful target | Runtime | Next tested failure |
|---|---:|---:|---:|
| AlphaFold 3 | EP300, 2,414 aa | 687 s | CEP350, 3,117 aa |
| Boltz-2 | None | — | CHRNA7, 502 aa |
| Chai-1 | None | — | NUP155, 1,391 aa |
| ESMFold2 | None | — | SLC26A8, 970 aa |
| ESMFold2 LM 300M | NUP155, 1,391 aa | 72 s | LCT, 1,927 aa |
| ESMFold2 LM 600M | NUP155, 1,391 aa | 92 s | LCT, 1,927 aa |
| IntelliFold2 | SLC26A8, 970 aa | 484 s | NUP155, 1,391 aa |
| OpenBind0 | LCT, 1,927 aa | 406 s | EP300, 2,414 aa |
| OpenDDE | None | — | CHRNA7, 502 aa |
| OpenFold3 | LCT, 1,927 aa | 425 s | EP300, 2,414 aa |
| Protenix2 | NUP155, 1,391 aa | 728 s | LCT, 1,927 aa |
| RoseTTAFold3 | None | — | CHRNA7, 502 aa |

Official AF3 was also tested on DMD (3,685 aa) and PRKDC (4,128 aa) with the v2 image at its 95% big-mode setting; both failed with GPU out-of-memory errors. Their allocation failures were 33.19 GiB and 45.79 GiB, respectively. CEP350 also failed at 90% (23.70 GiB requested), so the lower pool did not extend the observed AF3 boundary.

## v2 tuning checks

These are separate runtime profiles, not the image default. Results show why one global memory-pool setting does not suit every engine.

| Engine | 90% pool: largest pass | Next tested failure | Command buffers disabled: result |
|---|---:|---:|---|
| AlphaFold 3 | EP300, 2,414 aa | — | Not tested |
| Boltz-2 | CHRNA7, 502 aa | SLC26A8, 970 aa | CHRNA7, 502 aa passed |
| Chai-1 | NUP155, 1,391 aa | LCT, 1,927 aa | NUP155, 1,391 aa failed |
| ESMFold2 | SLC26A8, 970 aa | NUP155, 1,391 aa | SLC26A8, 970 aa failed |
| ESMFold2 LM 300M | NUP155, 1,391 aa | LCT, 1,927 aa | Not tested |
| ESMFold2 LM 600M | NUP155, 1,391 aa | LCT, 1,927 aa | Not tested |
| IntelliFold2 | SLC26A8, 970 aa | NUP155, 1,391 aa | Not tested |
| OpenBind0 | LCT, 1,927 aa | EP300, 2,414 aa | Not tested |
| OpenDDE | None | CHRNA7, 502 aa | CHRNA7, 502 aa failed |
| OpenFold3 | LCT, 1,927 aa | EP300, 2,414 aa | Not tested |
| Protenix2 | SLC26A8, 970 aa | NUP155, 1,391 aa | Not tested |
| RoseTTAFold3 | CHRNA7, 502 aa | SLC26A8, 970 aa | CHRNA7, 502 aa failed |

The 90% override allowed several engines to complete small-target jobs that failed at 95%, including Boltz-2, Chai-1, ESMFold2, and RoseTTAFold3. It reduced Protenix2’s observed limit from NUP155 at 95% to SLC26A8 at 90%. Disabling command buffers allowed the tested Boltz-2 CHRNA7 job to pass but did not fix the tested Chai-1, ESMFold2, OpenDDE, or RoseTTAFold3 failures. These are per-engine observations, not a universal recommended setting.

## Interpretation and limits

- The empirical v2 big-mode limit in this sweep ranges from no successful tested target for OpenDDE at 502 aa to EP300 at 2,414 aa for official AF3. Several AF3 clones passed LCT at 1,927 aa; Protenix2 and the ESMFold2 LM variants passed NUP155 at 1,391 aa.
- Failure means this exact image, engine, target, GPU, and workload did not complete. It is not proof that the model cannot run the sequence with other settings, hardware, or code.
- GPU memory values in the job records are allocator reservations/samples, not a precise peak of live model tensors. The optional TOKAMAX autotune `NameLoc.is_a_name` traceback was non-fatal; the runner continued with safe settings. OOM errors are separately logged.
- The toolkit’s two-GPU rowpair launcher was checked, but its strict source-hash compatibility guard rejected the installed AF3 tree. No two-GPU inference result is claimed.
- Only v2 results are used for the tables and conclusions above. The all-image CSVs are kept as an audit archive; v1 rows are not used to estimate v2 capability.

## Files

- `all-attempts.csv` and `capability-summary.csv`: v2-only primary results.
- `all-attempts-all-images-archive.csv` and `capability-summary-all-images-archive.csv`: historical audit archive across image tags.
- `v2-*.json`: manifests tying each prediction to its settings and app job ID.
- `rowpair-integration-checks.csv` / `.json`: two-GPU launcher compatibility checks (not folding runs).
- `gpu-telemetry-final-protenix-970.csv`: GPU sampling during the final Protenix2 970-aa validation.

Each prediction is also retained as an ordinary mn-cofolding job under the configured work directory and can be opened in the app by its job ID.
