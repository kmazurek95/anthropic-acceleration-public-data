# Data

- `benchmark_data_2026-11-13_wayback-20261113150458.zip`: Epoch AI, "Capabilities & benchmarking" data, as captured
  by the Wayback Machine on 13 November 2026 (capture `20261113150458`) from Epoch's download at
  https://epoch.ai/data/benchmark_data.zip. Fetch it from
  https://web.archive.org/web/20261113150458id_/https://epoch.ai/data/benchmark_data.zip
  SHA-256 `b07bd2eb34cc26469ea832daa2b17e6088f127caa3361c0ee78bac1d05671814`. Every script checks this hash before running.
- `input_manifest_2026-11-13.json`: retrieval time, hash and file inventory of the direct download described below,
  and the eci-public commit used (`e04929ab751531e558b1a61e86f436487c710ef8`).

**Why the archived copy.** The analysis was first run on a direct download from the same URL on the evening of
13 November (SHA-256 `66001629a0e32166db36c4d31e681137a3857f213658cd0d18e3dd287ea637c2`). That download is a later
build than the Wayback capture and cannot be fetched again, so every result here is computed from the Wayback capture.
The two differ in three score rows of the fit input (Claude Opus 4.7 and Mistral Large 4 Preview on MirrorCode,
Claude Fable 5.2 on SpireClimb). Epoch's published ECI scores and bootstrap files
(`epoch_capabilities_index/eci_scores.csv`, `eci_bootstraps.json`) are byte-identical between the two. The full
comparison is in [provenance/wayback_comparison/REPORT.md](../provenance/wayback_comparison/REPORT.md).

Epoch updates the file at that URL in place, so a fresh download is a newer snapshot and will fail the hash check.
Results from a newer snapshot will differ.

**License.** Epoch AI's data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Benchmark
questions and answers belong to their creators, and files from external projects (`*_external.csv`) keep their
original licenses. Citation: Epoch AI, "Capabilities & benchmarking". Published online at epoch.ai. Retrieved from
https://epoch.ai/benchmarks.
