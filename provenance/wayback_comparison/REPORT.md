# Epoch input: Wayback capture against direct download

*Last updated: 2026-11-16*

Two copies of Epoch's `benchmark_data.zip` from 13 November 2026 exist. Every result in this repository is computed
from the Wayback capture, because anyone can fetch it. This file records how the two copies differ. The comparison
was run by Claude Code on 16 November 2026.

## The two files

- **Wayback capture (the input for every result here):** `data/benchmark_data_2026-11-13_wayback-20261113150458.zip`
  - Fetch it from `https://web.archive.org/web/20261113150458id_/https://epoch.ai/data/benchmark_data.zip` (HTTP 200, 2,311,400 bytes).
  - SHA-256 `b07bd2eb34cc26469ea832daa2b17e6088f127caa3361c0ee78bac1d05671814`
- **Direct download (not used for any result):** `data/benchmark_data_2026-11-13.zip`
  - Downloaded from `https://epoch.ai/data/benchmark_data.zip` on the evening of 13 November. 2,311,563 bytes.
  - SHA-256 `66001629a0e32166db36c4d31e681137a3857f213658cd0d18e3dd287ea637c2`
  - Epoch updates that URL in place, so this copy cannot be fetched again.

Neither zip is distributed in this repository (see `data/README.md`).

## Internal timestamps

Zip entries carry a date and time with no time zone.

- Wayback capture: 90 entries stamped 2026-11-13 14:15:44. `benchmark_metadata.csv` and `model_metadata.csv` are stamped 14:15:46.
- Direct download: 90 entries stamped 2026-11-14 00:26:56. `benchmark_metadata.csv` and `model_metadata.csv` are stamped 00:26:58.

On these stamps the direct download is the later build.

## File-by-file comparison

- Both archives hold the same 92 file names. No file is present in only one of them.
- 85 files are byte-identical in content. Only their entry timestamps differ.
- 7 files differ in content: `bool_q_external.csv`, `gsm8k_external.csv`, `hella_swag_external.csv`, `mirrorcode.csv`, `model_metadata.csv`, `spireclimb.csv`, `spireclimb_sts1_a20_heart.csv`.
- Epoch's published index files are byte-identical: `epoch_capabilities_index/eci_scores.csv`, `eci_bootstraps.json`, `edi_scores.csv`, `processed_data_for_eci.csv`.
- In `bool_q_external.csv`, `gsm8k_external.csv` and `hella_swag_external.csv` the one changed line is BLOOM's country field, with the same two countries in a different order. Scores are unchanged.

## Effect on the fit input

The fit input is Epoch's loader (`eci.prepare_benchmark_data` at eci-public `e04929ab`) applied to each zip. It gives
256 models, 57 benchmarks and 2,879 rows from both. Three of the 2,879 score rows differ:

1. Claude Opus 4.7 on MirrorCode: 0.433333 (`claude-opus-4-7_max`) in the direct download, 0.311111 (`claude-opus-4-7_high`) in the Wayback capture. The direct download adds a `_max` row that the capture does not have.
2. Mistral Large 4 Preview on MirrorCode (`mistral-large-4-0_high`): 0.400000 in the direct download, 0.277778 in the Wayback capture.
3. Claude Fable 5.2 on SpireClimb (`claude-fable-5-2_max`): 0.154505 in the direct download, 0.196171 in the Wayback capture.

Refitting Epoch's prior (least-squares) method on each copy moves no model's score by more than 0.35 points. Results on
Epoch's published scores are the same for both copies, because `eci_scores.csv` is byte-identical.

## Model metadata

- The two copies of `model_metadata.csv` have 1,253 rows (direct) and 1,242 rows (Wayback).
- 17 rows differ, all for models that are not fitted: BLOOM-176B, Devstral Small, Gemini 4 Hydrogen, Muse Ember.
- For all 256 fitted models, the group table the pipeline builds (`accel.inputs.group_table`) is identical: release date, developer and accessibility match.

## Results

Every number in `RESULTS.md` comes from a full run on the Wayback capture.

A quick in-memory ratio check from before the rerun is kept in the private build repo; the full rerun supersedes it.

## Files here

- `cmpzip.py`: lists members and compares their content hashes. Output: `cmpzip_output.txt`.
- `cmpfit.py`: compares the loader output, the least-squares scores, the fitted-model group table and `model_metadata.csv`. Output: `cmpfit_output.txt`.

Both need the two zips in `data/`. Run them from the repository root with the virtual environment active:

```
python -I provenance/wayback_comparison/cmpzip.py data/benchmark_data_2026-11-13.zip data/benchmark_data_2026-11-13_wayback-20261113150458.zip
python -I provenance/wayback_comparison/cmpfit.py data/benchmark_data_2026-11-13.zip data/benchmark_data_2026-11-13_wayback-20261113150458.zip <absolute path to src>
```

The direct download cannot be fetched again, so for anyone without it the saved outputs are the record of the comparison.
