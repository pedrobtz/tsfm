# Golden parity fixtures

Each native architecture ships a small set of **golden fixtures**: fixed inputs
paired with the reference implementation's outputs. The parity tests
(`test-parity-*.R`) replay them and assert the R port matches within float
tolerance, so CI verifies correctness without needing Python, torch weights, or
Hub access at run time.

## Layout

```
fixtures/
├── ttm/
│   ├── case-01.json
│   └── ...
└── timesfm/
    ├── case-01.json
    └── ...
```

## Fixture schema (`*.json`)

```json
{
  "model_id": "ibm-granite/granite-timeseries-ttm-r2",
  "revision": "<pinned commit sha>",
  "context": [/* numeric history, oldest first */],
  "quantile_levels": [0.1, 0.5, 0.9],
  "expected_median": [/* h reference point forecasts */],
  "expected_quantiles": [[/* per-horizon quantile rows, optional */]],
  "tolerance": 1e-4
}
```

## Generating fixtures (once the numerical port is ready)

Fixtures are produced **outside CI**, in an environment with the Hub checkpoint,
torch, and the model's reference Python package (`granite-tsfm` for TTM,
`timesfm` for TimesFM), then committed:

1. Pin the checkpoint revision to a commit SHA (never a moving branch).
2. For each case, run the reference Python model on the `context` and record its
   median (and optionally full quantile) outputs.
3. Write one JSON file per case here, using the schema above.
4. Run `testthat::test_file("tests/testthat/test-parity-ttm.R")` with R torch to
   confirm the port reproduces them within `tolerance`.

Keep contexts short (tens of points) so the files stay small and the tests fast.
