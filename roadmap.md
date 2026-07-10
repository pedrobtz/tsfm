# `tsfm` Roadmap

A staged delivery plan for **`tsfm`**, a native R package that unifies
time-series foundation models (TSFMs) behind one tidymodels-first interface.
Derived from [`plan.md`](./plan.md); read that document for the full gap
analysis and rationale.

**North star:** one function to load any supported model from the Hugging Face
Hub, one function to forecast panels of one or many series, and one forecast
object that plugs into the existing R forecasting stack (tidymodels + tidyverts).

**Guiding principle:** tidymodels-first — a hardhat input contract, a parsnip
spec so models are swappable inside `workflow()`, and `tune`/`rsample`/
`yardstick` compatibility, with thin secondary adapters into fabletools/modeltime.

---

## Stage overview

| Stage | Name | Focus | Duration | Exit gate |
|-------|------|-------|----------|-----------|
| 0 | Foundations | Scaffold, contracts, and end-to-end plumbing on a stub | ~3 wks | Stub round-trips through tidy + fable APIs |
| 1 | First native model | TTM + Chronos-2 adapter + parsnip engine | ~4 wks | Workflow swaps models by `model_id` only |
| 2 | High-demand model + CRAN | Native TimesFM 2.5, hardened devices, tuning | ~5 wks | TimesFM parity; first CRAN submission |
| 3 | Best-in-class + covariates | Native Chronos-2, multivariate targets, conformal | ~5 wks | Chronos-2 parity incl. covariate paths |
| 4 | Fine-tuning & breadth | `tsfm_finetune()` (full + LoRA), Moirai-2, catalogue growth | post-1.0 | Fine-tuned TTM beats zero-shot on held-out data |

Each stage builds on the prior one; the input/output contracts are frozen in
Stage 0 so later stages add models without reworking the surface.

---

## Stage 0 — Foundations (M0)

Establish the package skeleton and every contract, proven end-to-end against a
trivial stub model so no architecture work is on the critical path yet.

**Deliverables**
- Package scaffold (`DESCRIPTION`, imports/suggests split, `on_load` hooks for
  optional tidymodels registration).
- Architecture **registry** (`registry.R`): maps an architecture string from
  `config.json` to a constructor + weight map.
- **Hub loader plumbing** (`hub.R`): `tsfm_pretrained()` — hfhub download,
  revision pinning, config parsing, safetensors → `nn_module` state dict —
  wired to a stub model.
- **hardhat bridge** (`bridge.R`): formula / recipe / data.frame / tsibble
  methods → a validated internal batch spec.
- **Preprocessing engine** (`preprocess.R`): shared composable steps driven by a
  declarative per-architecture *pipeline spec*, not per-model code.
- **Capability schema + pre-flight validation** (`capabilities.R`).
- **Forecast object** (`forecast.R`): `tsfm_forecast`, tidy `predict()`,
  `as_fable()`, `as_tibble()`, print/autoplot.

**Exit criterion**
- `fit -> predict -> yardstick::rmse()` works on the stub, **and**
- `forecast() |> as_fable() |> fabletools::accuracy()` works on the stub.

---

## Stage 1 — First native model (M1)

Prove the whole pipeline with a real, low-risk architecture and light up the
parsnip surface so foundation models become interchangeable.

**Why TTM first:** <1M params, pure MLP-mixer (no attention kernels → lowest
R-torch risk), Apache-2.0 weights (verified), multivariate + exogenous support,
seconds-fast on CPU — an ideal CPU-friendly default that validates the loader,
preprocessing, and forecast contracts end-to-end.

**Deliverables**
- Native **TTM** (`arch/ttm.R`) — `granite-timeseries-ttm-r2`.
- **Chronos-2 adapter** backed by `brulee`, behind the same
  `tsfm_pretrained("amazon/chronos-2")` call, so the uniform API covers
  Chronos-2 from day one.
- **Batched panel inference** (`batching.R`): key-batched, chunked, device-aware.
- **parsnip spec** `tsfm_reg()` + engine registration (`parsnip.R`).
- Vignette: *"Zero-shot forecasting in a workflow."*
- Vignette: *"Forecasting retail demand with covariates"* — TTM on a daily/weekly
  retail panel with `price`/`promo` past + known-future covariates
  (`tsibbledata::aus_retail`-style data), showcasing the exogenous-covariate
  path and multi-series batching.

**Exit criterion**
- TTM numerical-parity tests green against the reference Python implementation
  (golden fixtures), **and**
- a `workflow()` swaps TTM ↔ Chronos-2 by changing `model_id` only.

---

## Stage 2 — High-demand model + CRAN (M2)

Ship the most-requested missing model and take the package public.

**Deliverables**
- Native **TimesFM 2.5** (200M) — decoder-only patching transformer; exercises
  quantile heads and long contexts (first attention-based architecture).
- **Device handling hardened** across CPU / CUDA / MPS.
- **`tunable()` / `dials` parameters** (context length, quantile levels).
- Rolling-origin vignette with `tune_grid()` over `rsample::sliding_period()`
  (plus `modeltime.resample`).
- Vignette: *"Long-context forecasting of high-frequency demand"* — TimesFM 2.5
  on half-hourly electricity (`tsibbledata::vic_elec`), showcasing long context
  windows and intraday/weekly seasonality where more history helps.
- **First CRAN submission** (download-on-first-use with size message;
  `skip_on_cran()` for anything touching the Hub; parity tests off fixtures).

**Exit criterion**
- TimesFM 2.5 parity tests green, **and**
- CRAN submission passes checks.

**Action item:** confirm TimesFM 2.5 weight license from its model card before
shipping; surface it in `tsfm_capabilities()`.

---

## Stage 3 — Best-in-class model + covariates (M3)

Replace the interim adapter with a native implementation and complete the
covariate/multivariate story.

**Deliverables**
- Native **Chronos-2** (`arch/chronos2.R`), replacing the brulee-backed adapter —
  unlocking multivariate targets and future fine-tuning that brulee excludes.
- **Multivariate targets** end-to-end (declared by passing multiple target
  columns; capability check rejects unsupported models pre-flight).
- Conformal vignette with `conformalForecast`.
- Vignette: *"Hierarchical forecasting with calibrated quantiles"* — Chronos-2 on
  quarterly Australian tourism (`tsibble::tourism`), showcasing native quantile
  output flowing into `fabletools::reconcile()` for coherent hierarchical
  forecasts and prediction intervals.
- `as_modeltime_table()` adapter fully exercised.

**Exit criterion**
- Chronos-2 parity including covariate paths.

**Action item:** confirm Chronos-2 weight license from its model card; surface
it in `tsfm_capabilities()`.

---

## Stage 4 — Fine-tuning & breadth (v2, post-1.0)

Deliver the v2 headline (fine-tuning) and grow the catalogue to match/exceed
Darts (4 models) and skforecast (6 families).

**Deliverables**
- **`tsfm_finetune()`** — full + LoRA (LoRA as wrapped `nn_linear` layers in
  plain R torch), using R `torch`/`luz` training loops.
- **AMP / mixed precision** where R torch supports it.
- Native **Moirai-2**, then community-driven additions (**TiRex**,
  **PatchTST-FM**) via the third-party registration API.

**Exit criterion**
- Fine-tuned TTM beats zero-shot on a held-out benchmark.

---

## Example vignettes (one per headline capability)

Beyond the mechanics vignettes, three worked examples each demonstrate one
capability on a real, R-available dataset. They ship with the stage that lands
the model they use, and each example's input doubles as a golden-fixture case
for that model's parity tests.

| Vignette | Stage | Model | Capability shown | Dataset |
|----------|-------|-------|------------------|---------|
| Forecasting retail demand with covariates | 1 | TTM | past + known-future covariates, panel batching | retail (`tsibbledata::aus_retail`-style, with `price`/`promo`) |
| Long-context forecasting of high-frequency demand | 2 | TimesFM 2.5 | long context, intraday/weekly seasonality | half-hourly electricity (`tsibbledata::vic_elec`) |
| Hierarchical forecasting with calibrated quantiles | 3 | Chronos-2 | native quantiles → reconciliation | quarterly tourism (`tsibble::tourism`) |

## Cross-cutting concerns (every stage)

- **Numerical parity is the per-model acceptance gate:** same inputs →
  forecasts within float tolerance of the reference Python implementation,
  committed as golden fixtures so CI needs neither Python nor the Hub.
- **Frozen contracts:** the hardhat input contract and `tsfm_forecast` output
  contract are set in Stage 0 and must not churn as models are added.
- **Declarative preprocessing:** each architecture contributes a pipeline spec,
  not bespoke code — this is what keeps one package instead of four and holds
  per-model code thin.
- **Fine-tuning readiness from day one:** architectures are ordinary
  `nn_module`s with no frozen-weight shortcuts, so Stage 4 needs only a training
  loop, not rework.
- **Licensing surfaced in capabilities:** confirm each model's weight license
  before shipping it; the package never redistributes weights.

## Explicit non-goals (v1)

Reimplementing backtesting, metrics, reconciliation, or conformal methods
(adapters only); hosted-API models (nixtlar covers TimeGPT); training from
scratch; distributed/multi-GPU execution. R's surrounding forecasting
infrastructure is already strong — `tsfm` owns only the thin bridge in
(hardhat/parsnip) and the adapters out (fable/modeltime).

## Key risks tracked across stages

| Risk | Mitigation | Stage most exposed |
|------|-----------|--------------------|
| R torch operator gaps (fused attention) | Stage 1 model is attention-free; build attention from composable base ops first; file mlverse issues early | Stage 2 |
| Upstream model/config churn | Revision pinning by default; registry keyed on `(architecture, config version)` | All |
| Weight licensing | Verify per model card before shipping; surface in `tsfm_capabilities()` (TTM already verified Apache-2.0) | Stages 2–3 |
| CRAN download/test constraints | Download on first use with size message; parity tests off fixtures; `skip_on_cran()` for Hub access | Stage 2 |
| Long-term maintenance burden | Smallest viable catalogue + third-party registration API; declarative specs; propose to mlverse/tidymodels governance once Stage 2 lands | Stage 4+ |
