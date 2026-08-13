# tsfm

<!-- badges: start -->
[![R-CMD-check](https://github.com/pedrobtz/tsfm/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/pedrobtz/tsfm/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

**An inference engine for time-series foundation models, natively in R.**

The target is simple: given a supported pretrained checkpoint and a numeric
context, `tsfm` produces predictive quantiles.

R already has every layer a foundation-model forecasting stack needs — Hub
access (`hfhub`), checkpoint format (`safetensors`), tensor runtime (`torch`),
temporal structures (`tsibble`), forecast distributions (`distributional`),
backtesting (`rsample`), reconciliation and metrics (`fabletools`), and conformal
calibration (`conformalForecast`). What it lacks is the **model layer**: a
catalogue of natively implemented time-series foundation models (TSFMs) behind
one capability-aware interface. `tsfm` is building that layer:

- **One loader.** `tsfm_pretrained("...")` resolves a supported model into a
  uniform handle. The current executable model is the weight-free `stub` test
  fixture; the first native release target is TimesFM 2.5.
- **One contract.** Every architecture implements
  `predict_fn(context, h, quantile_levels)` and nothing more. See
  `` ?`tsfm-architecture-contract` `` and verify yours with
  `tsfm_check_architecture()`.
- **Capability metadata, checked first.** `tsfm_capabilities()` reports context
  and horizon limits, trained quantile levels, and the weight licence. Contract
  v1 fixes multivariate, covariate, sample, and fine-tuning flags at `FALSE`
  until those inputs and outputs have real execution channels.
- **An offline checkpoint catalogue.** `tsfm_models()` lists only checkpoints
  that passed the support gates; `tsfm_models(state = NULL)` also exposes
  experimental and scaffold records. Listing only probes the local Hub cache
  and never downloads anything.
- **One forecast object** backed by `distributional`.

### Framework-neutral by design

The engine has no opinion about where your context came from or where the
quantiles go. Interfaces to the surrounding ecosystem are **optional adapters of
equal standing**, all in `Suggests`:

| Adapter | Surface | Needs |
|---------|---------|-------|
| plain R | `forecast()` on a `data.frame`, `as.data.frame()` | — |
| tidyverts | `fabletools::as_fable()` → `fable` | `fabletools`, `tsibble` |
| tidymodels | `tsfm_fit()` / `predict()`, `tsfm_reg()` parsnip spec | `hardhat`, `parsnip` |

`tsfm` deliberately does **not** define competing `forecast()` or `as_fable()`
generics: it re-exports `generics::forecast` and registers its `as_fable` method
onto `fabletools`, so attaching `tsfm` alongside them never masks the verbs.

## Installation

```r
# development version
# install.packages("pak")
pak::pak("pedrobtz/tsfm")
```

The current development baseline exercises forecasting without downloading
weights. The checkpoint pipeline itself is now available: explicit prefetch,
safetensors validation/loading, disk/resident cache inspection, and bounded
constructed-handle reuse. TimesFM inference remains blocked until the native
module passes the committed golden fixtures.

## Quick start

```r
library(tsfm)

history <- data.frame(
  time = 1:24,
  value = 100 + cumsum(sin((1:24) * pi / 6))
)

# Weight-free random-walk fixture for exercising the engine shell.
model <- tsfm_pretrained("stub")
tsfm_capabilities(model)

# Plain-R engine path.
fc <- forecast(model, history, h = 6, index = "time", target = "value")
as.data.frame(fc)

# Optional tidyverts adapter.
fc |> fabletools::as_fable()
```

See `vignette("zero-shot-workflow")` and `vignette("rolling-origin-tuning")`.

## Adding an architecture

Third parties extend the catalogue without forking. Implement the contract,
verify it, register it:

```r
my_arch <- function(config, weights) {
  new_tsfm_model(
    architecture = "my-arch",
    config       = config,
    capabilities = new_tsfm_capabilities("my-arch", max_context = 512L),
    predict_fn   = function(context, h, quantile_levels) {
      # ... -> matrix[h, length(quantile_levels)]
    }
  )
}

tsfm_check_architecture(my_arch)          # contract conformance gate
tsfm_register_arch("my-arch", my_arch)    # from your own .onLoad()
```

## Model catalogue

The programmatic catalogue is the source of truth:

```r
tsfm_models() # safe default: supported checkpoints only
```

| Model | id | Status |
|-------|----|--------|
| Stub | `stub` | executable random-walk test fixture; not a foundation model |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | registered scaffold; forward pass not implemented |
| TTM (TinyTimeMixer) | `ibm-granite/granite-timeseries-ttm-r2` | registered scaffold; deferred until point-only output is supported |
| Chronos-2 | `amazon/chronos-2` | unregistered reference adapter; not supported in `0.1.0` |

No real foundation model is supported yet. “Supported” will require both the
architecture conformance gate and numerical parity against a pinned reference.

Third parties can register additional architectures with
`tsfm_register_arch()` — no fork required.

### Checkpoint provenance, cache, size, and licence

The TimesFM feasibility work is pinned to official source commit
`3dae50b20d7a724981e8ea36cda75578f80dd2dc` and checkpoint revision
`1d952420fba87f3c6dee4f240de0f1a0fbc790e3`. The catalogue never resolves a
moving `main` branch for curated support metadata. TimesFM remains a
`scaffold`: `tsfm_pretrained()` rejects it before Hub or tensor work until the
full loader and numerical-parity gates pass.

`cached` means every file in the checkpoint's static manifest is already in
the cache managed by `hfhub`. Probes always set `local_files_only = TRUE`.
`hfhub` uses `HUGGINGFACE_HUB_CACHE` when set and otherwise defaults to
`~/.cache/huggingface/hub`; `tsfm` does not create a second disk cache. A known
incomplete manifest is `FALSE`, while `NA` means the catalogue entry has no
manifest definition yet.

`size_bytes` is a static first-download estimate exposed before loading. The
pinned TimesFM `model.safetensors` is 925,181,104 bytes (about 925 MB, or 882
MiB), excluding small metadata files and live tensor/runtime overhead. The
catalogue's `license` column is the upstream **weight** licence—Apache-2.0 for
the two current scaffold records—not the package's MIT licence. `tsfm` reports
the upstream terms and never accepts gated-repository terms on a user's behalf.

### Download and constructed-handle lifecycle

Disk files and live R objects are deliberately separate:

```r
tsfm_cache_status()
tsfm_unload() # releases resident handles; does not delete Hub files
```

`tsfm_download(model_id, revision = NULL)` explicitly fetches a curated
checkpoint manifest and returns its local paths invisibly without constructing
a module. It may be used to stage a scaffold checkpoint for development, but
`tsfm_pretrained()` still rejects non-supported catalogue states before any
download or tensor work.

Constructed handles use an R-session least-recently-used cache keyed by model
ID, immutable revision, resolved device, and every load-affecting option.
`options(tsfm.max_loaded_models = 1L)` is the default; `0L` disables resident
reuse. `reuse = FALSE` constructs a fresh handle without changing the existing
cached handle or deleting downloaded files.

## Scope

`tsfm` owns the model layer and thin adapters only. Backtesting, metrics,
reconciliation, conformal calibration, tuning, and feature engineering are
delegated to the existing R stack.

Explicitly **out of scope**: hosted-API models (`nixtlar` covers TimeGPT),
training from scratch, and distributed execution. `tsfm` runs open weights
locally; anything served over a network belongs to its own client.

See [`.agents/roadmap.md`](.agents/roadmap.md) for the staged plan and
[`.agents/plan.md`](.agents/plan.md) for
the full gap analysis.
