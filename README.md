# tsfm

<!-- badges: start -->
[![R-CMD-check](https://github.com/pedrobtz/tsfm/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/pedrobtz/tsfm/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

**Native time-series foundation models for R**, behind one uniform,
tidymodels-first interface.

R already has every layer a foundation-model forecasting stack needs — Hub
access (`hfhub`), checkpoint format (`safetensors`), tensor runtime (`torch`),
temporal structures (`tsibble`), forecast distributions (`distributional`),
backtesting (`rsample`), reconciliation and metrics (`fabletools`), and conformal
calibration (`conformalForecast`). What it lacks is the **model layer**: a
catalogue of natively implemented time-series foundation models (TSFMs) behind
one capability-aware interface. `tsfm` fills that gap:

- **One loader.** `tsfm_pretrained("...")` resolves any supported model from the
  Hugging Face Hub (revision pinned) into a uniform handle.
- **One forecast surface.** `forecast()` for `tsibble`/data-frame panels, or a
  `parsnip` engine (`tsfm_reg()`) so models drop into `workflows`, `tune`, and
  `rsample` unchanged. Swap models by changing a single `model_id`.
- **Capability metadata, checked first.** `tsfm_capabilities()` reports context
  limits, covariate/multivariate support, quantiles, and licence — and requests
  that exceed them fail fast, before any tensor work.
- **One forecast object** backed by `distributional`, with near-free adapters to
  `fable` and `modeltime`.

## Installation

```r
# development version
# install.packages("pak")
pak::pak("pedrobtz/tsfm")
```

`torch` provides the runtime; the first use of a model downloads its weights
from the Hub (with an explicit size message) and caches them.

## Quick start

```r
library(tsfm)

model <- tsfm_pretrained("amazon/chronos-2")
tsfm_capabilities(model)

# tidymodels
fit <- tsfm_fit(demand ~ date, data = train_df, model = model,
                index = "date", id = "store")
predict(fit, new_data = future_df)   # .pred, .pred_lower, .pred_upper, ...

# or the tsibble convenience path
fc <- forecast(model, history_tsbl, h = 12)
fc |> as_fable() |> fabletools::accuracy(actuals_tsbl)
```

See `vignette("zero-shot-workflow")` and `vignette("rolling-origin-tuning")`.

## Model catalogue

| Model | id | Status |
|-------|----|--------|
| Chronos-2 | `amazon/chronos-2` | available (via a `brulee`-backed adapter) |
| TTM (TinyTimeMixer) | `ibm-granite/granite-timeseries-ttm-r2` | native port in progress |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | native port in progress |

Third parties can register additional architectures with
`tsfm_register_arch()` — no fork required.

## Scope

`tsfm` owns the model layer and thin adapters only. Backtesting, metrics,
reconciliation, conformal calibration, tuning, and feature engineering are
delegated to the existing R stack. See [`roadmap.md`](roadmap.md) for the staged
plan and [`plan.md`](plan.md) for the full gap analysis.
