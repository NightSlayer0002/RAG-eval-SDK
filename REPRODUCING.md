# Reproducing the frozen experiments

Run commands from the repository root after installing `python -m pip install -e ".[dev]"`. Offline tests and saved-prediction replay require no API key, model download, or generation calls.

## Obtain the original dataset

Download `source_info.jsonl` and `response.jsonl` from the original [RAGTruth repository](https://github.com/ParticleMedia/RAGTruth). Put them in `benchmarks/data/RAGTruth/`, or pass another directory to the replay functions below. Raw datasets are ignored by Git.

The audits verify these SHA-256 values before calculating results:

| File | SHA-256 |
|---|---|
| source_info.jsonl | `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b` |
| response.jsonl | `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073` |

A different upstream revision may fail these checks. Do not weaken the checks or substitute a different dataset to claim exact replay.

## Replay without changing frozen files

Run this Python code from the repository root:

```python
from pathlib import Path
from benchmarks.audit_frozen_run import audit as audit_step1
from benchmarks.audit_step2_spans import audit as audit_step2

data = Path("benchmarks/data/RAGTruth")
results = Path("benchmarks/results")

step1 = audit_step1(results / "ragtruth_mreg_hhem_ragas_fresh100", data)
step2 = audit_step2(results / "ragtruth_step2_spans_frozen_20260821", data)
assert step1["status"] == step2["status"] == "pass"
print(step1)
print(step2)
```

These functions reconstruct cohort membership, source groups, labels, thresholds, summaries and paired comparisons from the original dataset and saved scores. Step 1 also refits the compact risk model; Step 2 reconstructs exact gold/predicted spans and its 5,000-resample source-group bootstrap. This validates saved-output reproducibility, not a new model-inference replication.

Historical manifests and audit reports retain their original Windows paths as provenance. The Step 2 audit resolves `benchmarks/...` exclusion checkpoints within the current checkout and validates their bytes against the original manifest hashes. Historical hash keys are preserved. The Step 1 source scan inspects the audit module's repository, independently of the dataset location.

The `python -m benchmarks.audit_frozen_run` and `python -m benchmarks.audit_step2_spans` CLI commands also replay results, but write `integrity_audit.json` into their output directory. Use the functions above to preserve frozen files byte for byte.

## Rerun inference separately

Install the relevant `[hhem]` and `[benchmark]` extras. The exact Step 1 command is preserved in [FROZEN_STEP1_REPORT.md](FROZEN_STEP1_REPORT.md); Step 2 parameters are preserved in [STEP2_PROTOCOL.md](STEP2_PROTOCOL.md) and its frozen manifest. Model and dataset pins are in [VALIDATION_STATUS.md](VALIDATION_STATUS.md).

Use a **new output directory** for inference. Do not overwrite published results. The manifests record historical SDK versions (`2.0.0a1` and `2.0.0a2`); this curated checkout contains the later `2.0.0a2` code plus audit portability fixes. It does not include an archived Step 1 source checkout or a complete dependency lock. Re-running inference is therefore not guaranteed to reproduce saved floating-point scores exactly. Remote judge availability and behavior can also change.

Supply your own credentials using the variable names in `.env.example`. Keep credentials outside Git. Local HHEM runs download model weights; enabling `trust_remote_code` is explicit in the historical configuration. The selected Ragas judge was `gemini-3.5-flash-lite`, using real Ragas Faithfulness 0.4.3.

Frozen test groups are consumed: use fresh groups and a new preregistration to evaluate a revised method.

## Test and build

```bash
python -m pytest -q
python -m pip wheel . --no-deps --wheel-dir dist
```

Only source packages go in the wheel. The GitHub repository also includes tests, reports and frozen evidence. Build outputs and environments are ignored.

## CLI input examples

`data/chat.json` and `data/context.json` contain fictional library examples. From a source checkout, `rag-eval run` uses these defaults and requires a configured generation provider. From an installed wheel, pass `--chat`, `--context`, and `--output` explicitly because repository examples are not packaged. The README Python example runs entirely offline.

## What was removed for GitHub

- Duplicate pre-SDK `src/` code and old pseudo-Ragas benchmark scripts/charts.
- Development pilots, smoke outputs, generated reports and caches; both frozen experiment bundles remain intact.
- Personal interview/study/market notes, obsolete guides and draft paper text.
- Real `.env` secrets, copied virtual environments, bytecode and package build metadata.
- Bulk datasets and model weights; obtain these separately from their owners.

The original project is retained separately. Historical reports describe some development runs whose artifacts are intentionally not included in this curated repository.

