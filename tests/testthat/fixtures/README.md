# Golden parity fixtures

Each native architecture ships a small set of **golden fixtures**: fixed inputs
paired with the reference implementation's outputs. The parity tests
(`test-parity-*.R`) replay them and assert the R port matches within float
tolerance, so CI verifies correctness without needing Python, torch weights, or
Hub access at run time.

## Layout

```
fixtures/
└── timesfm/
    ├── typical.json
    ├── short_context.json
    ├── context_truncation.json
    └── batch_agreement.json
```

## Fixture schema (`*.json`)

```json
{
  "schema_version": 1,
  "name": "typical",
  "source_commit": "<pinned source sha>",
  "model_id": "google/timesfm-2.5-200m-pytorch",
  "revision": "<pinned checkpoint sha>",
  "reference": {"torch": "2.2.2", "...": "forecast flags"},
  "context_specs": [{"kind": "trend_seasonal", "length": 96}],
  "max_context": 96,
  "horizon": 12,
  "quantile_levels": [0.1, 0.2, "...", 0.9],
  "expected_point": [[/* series by horizon */]],
  "expected_quantiles": [[[/* series by horizon by quantile */]]],
  "tolerance": 1e-4
}
```

Long contexts use a deterministic generator descriptor instead of embedding
thousands of values, keeping the committed fixtures small.

## Regenerating the TimesFM fixtures

The generator is `.agents/generate-timesfm-reference.py`; its Python packages
are locked in `.agents/timesfm-reference-requirements.txt`. It refuses to run
unless `TIMESFM_SOURCE` is exactly the recorded source commit.

```sh
python -m venv .venv-timesfm-reference
.venv-timesfm-reference/bin/pip install \
  -r .agents/timesfm-reference-requirements.txt
TIMESFM_SOURCE=/path/to/pinned/timesfm \
TIMESFM_CHECKPOINT=/path/to/model.safetensors \
  .venv-timesfm-reference/bin/python \
  .agents/generate-timesfm-reference.py \
  tests/testthat/fixtures/timesfm
```

The two-series fixture also evaluates each context separately and records the
maximum batch/loop difference. Fixture generation is outside CI; replaying the
committed outputs in the eventual native parity test is network- and
Python-free.
