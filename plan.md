# A Native R Package for Time-Series Foundation Models: Review, Verification, and Plan

*Prepared 2026-07-10. All external claims re-verified against current sources on this date.*

---

## 1. Executive summary

The submitted 22-point gap analysis is **substantially correct**. Re-running the
investigation confirmed every load-bearing claim, with only minor refinements
(§2). The R ecosystem has all the *infrastructure* layers a foundation-model
forecasting stack needs — Hub access (`hfhub`), checkpoint format
(`safetensors`), tensor runtime (`torch`), temporal data structures
(`tsibble`), forecast distributions (`distributional`/`fabletools`),
backtesting (`rsample`), reconciliation (`fabletools`), and conformal
calibration (`conformalForecast`) — but it is missing the **model layer**: a
catalogue of natively implemented time-series foundation models (TSFMs) behind
one uniform, capability-aware interface. That layer is exactly what Darts'
`FoundationModel` collection (4 models) and skforecast's
`FoundationModel`/`ForecasterFoundation` (6 model families) now provide in
Python, and what only `brulee::brulee_chronos()` provides in R — for a single
model, inference-only.

This document proposes a single new package, working name **`tsfm`**, that
covers that gap: a `from_pretrained()`-style loader with an architecture
registry, native R `torch` implementations of a curated model catalogue
(TTM → TimesFM 2.5 → Chronos-2 → Moirai-2), a shared preprocessing contract,
explicit capability metadata, batched panel inference, and a standard forecast
object. It is designed **tidymodels-first**: hardhat-based `formula` /
`data.frame` / `recipe` interfaces (the exact pattern `brulee_chronos()`
already uses), a parsnip model specification so the models drop into
`workflows`, `tune`, and `rsample` unchanged, and yardstick-compatible
output — with thin secondary adapters into fabletools/modeltime for the
tidyverts world. Fine-tuning is designed for from day one but scheduled as
the v2 headline.

---

## 2. Review of the analysis: verification results

Each claim that the analysis's conclusion rests on was independently
re-checked (2026-07-10). Verdicts:

| # | Claim in analysis | Verdict | Notes from re-run |
|---|---|---|---|
| 1 | `hfhub` downloads files/snapshots with HF-compatible cache | ✅ Confirmed, one caveat | On CRAN (v0.1.1). The CRAN README documents `hub_download()` as the primary export; snapshot support should be re-confirmed against the dev version before relying on it. Revision pinning works (brulee uses a pinned commit SHA through this path). |
| 2 | `safetensors` is a pure-R reader/writer feeding R `torch` | ✅ Confirmed | On CRAN; used in production by `brulee`'s Chronos-2 loader. |
| 3 | R `torch` provides the runtime (CPU/CUDA/MPS) | ✅ Confirmed | Known limitation stands: smaller custom-kernel ecosystem than PyTorch. |
| 4 | No AutoModel-style registry in R | ✅ Confirmed — with a framing refinement | See §2.1 below: the gap is real but "AutoModel" slightly misframes it. |
| 5 | `brulee::brulee_chronos()` is the only native local TSFM; no native TimesFM/MOIRAI/TTM/TiRex | ✅ Confirmed in detail | Source inspection of `tidymodels/brulee` (`R/chronos2-*.R`): downloads `amazon/chronos-2` via pinned SHA to `~/.cache/chronos-r/` (~500 MB), loads safetensors into R torch, **frozen weights only**, exactly one numeric target, past covariates (+ known-future values via `new_data`), multi-series via `id_column`, quantile output (default deciles). No other native TSFM found for any architecture on CRAN or GitHub. |
| 6 | No uniform cross-model API in R; Darts standardizes 4 TSFMs, skforecast 6 families | ✅ Confirmed | Darts `FoundationModel`: Chronos-2, TimesFM 2.5, TiRex, PatchTST-FM (Moirai *not* included). skforecast: Chronos-2, TimesFM 2.5, Moirai-2, TabICL, TabPFN-TS, TFC T0. No R equivalent exists or is announced. |
| 7 | No `model_capabilities()`-style pre-flight validation in R | ✅ Confirmed | Nothing found. |
| 8 | `tsibble` is a strong native panel/temporal structure | ✅ Confirmed | No change. |
| 9 | No reusable TSFM preprocessing contract | ✅ Confirmed | brulee's pipeline is internal to `brulee_chronos()` and Chronos-2-specific. |
| 10–14 | Partial-coverage assessments (batching, multivariate, covariates, zero-shot, quantiles) | ✅ Confirmed | brulee source inspection matches every stated limitation, including single-target-only (item 11) and frozen weights (item 15). `nixtlar` (CRAN, v0.6.x) is API-hosted TimeGPT only. |
| 15–16 | No fine-tuning / PEFT / mixed-precision TSFM stack in R | ✅ Confirmed | brulee docs explicitly state "no training is performed; the network has fixed pretrained weights." |
| 17–22 | R strong on backtesting, metrics, conformal, reconciliation; gap is TSFM integration | ✅ Confirmed | `conformalForecast` reached CRAN 2026-01-15 (classical/adaptive/PID/AcMCP). `fabletools` reconciliation unchanged. |

### 2.1 Refinements to the analysis

1. **The "AutoModel" framing (item 4) slightly overstates what Python has.**
   Most Python TSFMs do *not* route through `transformers.AutoModel` — Chronos,
   TimesFM, and uni2ts/Moirai each ship their own standalone loader. What
   Python actually has is *per-model native libraries plus two unifying
   wrappers* (Darts, skforecast). The R gap is therefore better stated as:
   **no native architecture implementations to dispatch to**, not the absence
   of a dispatch mechanism per se. The registry is a week of work; the model
   implementations behind it are the real effort. This matters for scoping §4.

2. **The analysis undersells the strongest evidence of feasibility.**
   `brulee_chronos()` is not just a partial coverage point — it is an
   end-to-end existence proof that the full pipeline (pinned Hub download →
   safetensors → R torch reimplementation of a modern TSFM → quantile
   forecasts, CPU/GPU) works natively in R with acceptable performance, and it
   is a design template the new package should copy deliberately. Likewise,
   mlverse's `minhub` (native R torch ports of GPT-2-class LLMs loaded from HF
   checkpoints) is prior art for the porting workflow.

3. **Item 20's "missing element" and items 17–19's "adapter needed" are the
   same deliverable.** A standard forecast object that speaks
   `fabletools`/`distributional` collapses roughly five of the partial-gap rows
   (17, 18, 19, 20, 21) into one thin adapter layer. The plan exploits this.

4. **Minor**: hfhub's snapshot capability should be treated as "verify before
   depending on it" (CRAN version documents single-file download as the
   primary API); and Darts standardizes TiRex/PatchTST-FM but *not* MOIRAI,
   which the analysis's item 6 could be read as implying.

**Overall review verdict:** the analysis's conclusion — that the consequential
gap is the model layer (items 4, 5, 6, 9, 11, 15) and not the surrounding
forecasting infrastructure — is correct and is adopted as the premise of the
plan below.

---

## 3. The main gap, stated precisely

> R can download any TSFM checkpoint and has the tensor runtime to execute it,
> but — outside of one frozen, single-target Chronos-2 implementation — it has
> **no native model implementations, no loader that turns a Hub config into an
> R torch module, no shared preprocessing contract, no capability metadata,
> and no uniform forecasting interface across models.**

One package can close this because the sub-gaps are one coherent layer: the R
equivalent of *(chronos-forecasting + timesfm + granite-tsfm)* unified the way
*Darts FoundationModel* unifies them.

---

## 4. Plan: the `tsfm` package

### 4.1 Mission and non-goals

**Mission.** Zero-shot (and later fine-tuned) forecasting with open
time-series foundation models, natively in R: one function to load any
supported model from the Hugging Face Hub, one function to forecast panels of
one or many series, one forecast object that plugs into the existing R
forecasting stack.

**Guiding principle: tidymodels-first.** The package should feel like a
tidymodels extension, not a parallel universe. Concretely:

- **hardhat input contract.** The user-facing fit/forecast interface follows
  the `brulee` pattern exactly — an S3 generic with `formula`, `data.frame`,
  `recipe`, and `matrix` methods built on `hardhat::mold()`/`forge()`, so
  covariate roles, factor handling, and novel-level checks behave identically
  to every other tidymodels engine. (`brulee_chronos()` already proves this
  pattern works for a TSFM; we adopt it rather than invent a new contract.)
- **parsnip model specification.** A `tsfm_reg()` spec (or engines registered
  under modeltime's existing time-series parsnip specs) with
  `set_engine("tsfm", model_id = "google/timesfm-2.5-200m-pytorch")`, so
  foundation models are exchangeable inside `workflow()` by changing one
  engine argument — the R analogue of Darts' "change only the model name".
- **recipes for covariates.** Exogenous features are prepared with standard
  `recipes` steps; the package contributes only what is TSFM-specific and not
  expressible as a recipe step (context truncation, patching, instance
  normalization, masks), which stays inside the model layer per §4.3.
- **tune/rsample/yardstick compatibility.** Tunable arguments (context
  length, quantile levels, and in v2 fine-tuning hyperparameters) are
  declared via `tunable()`/`dials` so `tune_grid()` over
  `rsample::sliding_period()` resamples works out of the box, scored by
  yardstick metrics.

tsibble input and fable adapters remain supported (§4.2), but as a secondary
surface: the canonical training/tuning path is data frame + recipe +
workflow.

**Explicit non-goals** (v1): reimplementing backtesting, metrics,
reconciliation, or conformal methods (adapters only — R already has these);
hosted-API models (nixtlar covers TimeGPT); training from scratch;
distributed/multi-GPU execution.

### 4.2 User-facing API (design sketch)

```r
library(tsfm)

# 4. Architecture registry + from_pretrained() loader  ── gap items 4, 5
model <- tsfm_pretrained("ibm-granite/granite-timeseries-ttm-r2",
                         revision = "main")   # hfhub download, pinned + cached
model <- tsfm_pretrained("google/timesfm-2.5-200m-pytorch")
model <- tsfm_pretrained("amazon/chronos-2")

# 7. Capability metadata, checked *before* execution   ── gap item 7
tsfm_capabilities(model)
#> <tsfm_capabilities>
#>   architecture:      ttm
#>   max_context:       1536        quantiles: native
#>   multivariate:      TRUE        samples:   no
#>   past_covariates:   TRUE        future_covariates: TRUE
#>   static_covariates: FALSE       fine_tunable: TRUE
#>   license:           Apache-2.0

# 6/9–12. Uniform, tidymodels-native fit interface (hardhat, brulee-style)
fit <- tsfm_reg_fit(demand ~ price + promo + date, data = train_df,
                    model = model, id_column = "store")
predict(fit, new_data = future_df)   # tibble: .pred, .pred_lower, .pred_upper, ...

# ... or as a parsnip spec inside a workflow — models are exchangeable
# by editing one engine argument (the R analogue of Darts FoundationModel):
spec <- tsfm_reg(context_length = tune()) |>
  set_engine("tsfm", model_id = "ibm-granite/granite-timeseries-ttm-r2")

wf <- workflow() |>
  add_recipe(recipe(demand ~ ., data = train_df) |> step_dummy(all_nominal())) |>
  add_model(spec)

res <- tune_grid(wf, resamples = sliding_period(train_df, date, "month"),
                 metrics = metric_set(rmse, quantile_loss))

# Convenience path for tsibble users (secondary surface)
fc <- forecast(model, tourism_tsb, h = 12,
               quantile_levels = c(0.1, 0.5, 0.9))

# 14 + 17–21. Standard forecast object with adapters
as_fable(fc)                        # -> fabletools accuracy / reconciliation
as_modeltime_table(fc)              # -> modeltime workflows

# v2: fine-tuning                                       ── gap items 15, 16
tuned <- tsfm_finetune(model, train_df, method = "lora", epochs = 5)
```

Design rules:

- **Input contract**: hardhat-molded data frames are canonical (formula,
  recipe, or x/y — identical to other tidymodels engines), with the time
  index and series keys declared as arguments or recipe roles; a `tsibble`
  method carries index/key metadata automatically. Multivariate targets are
  declared by passing multiple target columns — the capability check rejects
  this for models that can't do it *before* any tensor work.
- **Output contract**: two views of the same result. `predict()` on a fitted
  object returns a tidymodels-conformant tibble (row-aligned with `new_data`,
  `.pred` / `.pred_lower` / `.pred_upper` and per-quantile columns) so
  yardstick and tune consume it directly; the richer `tsfm_forecast` object
  holds, per key × horizon, whichever of {native quantiles, sample paths,
  parametric distribution} the model produces, stored via
  `distributional::dist_*` vectors so `as_fable()` is nearly free.
- **Everything is generic-first** (`forecast()`, `predict()`,
  `tsfm_capabilities()`) so third parties can register additional
  architectures without touching the package.

### 4.3 Internal architecture

```
tsfm/
├── R/
│   ├── registry.R        # arch string (from config.json) -> constructor + weight map
│   ├── hub.R             # tsfm_pretrained(): hfhub download, revision pinning,
│   │                     #   config parsing, safetensors -> nn_module state dict
│   ├── bridge.R          # hardhat mold/forge bridge: formula/recipe/data.frame/
│   │                     #   tsibble methods -> validated internal batch spec
│   ├── parsnip.R         # tsfm_reg() spec, engine registration, tunable()/dials
│   │                     #   parameters (context_length, quantile_levels, ...)
│   ├── preprocess.R      # shared, composable steps: sort/regularize, gap+NA masks,
│   │                     #   scaling (per-series/instance norm), context truncation,
│   │                     #   padding, patching, frequency encoding
│   │                     #   Each architecture declares a *pipeline spec*, not code.
│   ├── capabilities.R    # per-model metadata schema + pre-flight validation
│   ├── batching.R        # key-batched inference, chunking, device placement
│   ├── forecast.R        # tsfm_forecast object, print/autoplot,
│   │                     #   as_fable(), as_modeltime_table(), as_tibble()
│   └── arch/
│       ├── ttm.R         # native torch: TinyTimeMixer (M1)
│       ├── timesfm.R     # native torch: TimesFM 2.5 decoder (M2)
│       ├── chronos2.R    # native torch: Chronos-2 (M3)
│       └── moirai.R      # native torch: Moirai-2 (M4, post-1.0)
```

- **Imports**: `torch`, `hfhub`, `safetensors`, `hardhat`, `distributional`,
  `rlang`, `cli`. **Suggests**: `parsnip`, `workflows`, `tune`, `dials`,
  `recipes`, `rsample`, `yardstick`, `tsibble`, `fabletools`, `fable`,
  `modeltime`, `conformalForecast`, `ggplot2`. (parsnip registration lives
  behind `on_load` hooks so the core package stays light, the same way
  brulee/modeltime engines register.)
- The preprocessing layer (gap item 9) is the piece that keeps this one
  package instead of four: each architecture contributes a declarative spec
  (patch length, scaling type, context limit, mask convention); the shared
  engine executes it. This mirrors GluonTS transformations, at much smaller
  scope.
- Fine-tuning readiness: architectures are ordinary `nn_module`s with no
  frozen-weight shortcuts baked in, so v2 fine-tuning (full + LoRA — LoRA is
  implementable in plain R torch as wrapped `nn_linear` layers) requires no
  rework, only a training loop, which R `torch`/`luz` already provides.

### 4.4 Model roadmap and rationale

| Milestone | Model | Why this order |
|---|---|---|
| **M1** | **TTM (granite-timeseries-ttm-r2)** | <1 M parameters, pure MLP-mixer (no attention kernels — lowest R-torch risk), Apache-2.0 weights (verified), multivariate + exogenous support, seconds-fast on CPU. Ideal to validate the whole loader/preprocess/forecast contract end-to-end and to give the package a CPU-friendly default model. |
| **M2** | **TimesFM 2.5 (200 M)** | Highest-demand missing model; decoder-only patching transformer; PyTorch safetensors checkpoint on the Hub; exercises quantile heads and long contexts. |
| **M3** | **Chronos-2** | Best model, but lowest marginal urgency because brulee already covers zero-shot. Native implementation unlocks multivariate targets and future fine-tuning that brulee deliberately excludes. Interim: an optional brulee-backed adapter behind the same `tsfm_pretrained("amazon/chronos-2")` call, so the uniform API covers Chronos-2 from M1. |
| **M4 (post-1.0)** | **Moirai-2**, then community-driven (TiRex, PatchTST-FM) | Matches/exceeds Darts (4) and skforecast (6 incl. tabular crossovers) catalogue breadth. |

Per-model acceptance gate: **numerical parity with the reference Python
implementation** on golden fixtures (same inputs → forecasts within float
tolerance), committed as test data so CI never needs Python or the Hub.

### 4.5 Milestones

| Milestone | Deliverables | Exit criterion |
|---|---|---|
| **M0** (~3 wks) | Package scaffold; hardhat bridge (formula/recipe/data.frame/tsibble); preprocessing engine; capability schema; `tsfm_forecast` object + tidy `predict()` + `as_fable()`; registry + `tsfm_pretrained()` plumbing on a stub model | `fit -> predict -> yardstick::rmse()` and `forecast() |> as_fable() |> fabletools::accuracy()` both work on the stub |
| **M1** (~4 wks) | Native TTM; brulee-backed Chronos-2 adapter; batched panel inference; `tsfm_reg()` parsnip spec + engine registration; vignette "Zero-shot forecasting in a workflow" | TTM parity tests green; a `workflow()` swaps TTM ↔ Chronos-2 by changing `model_id` only |
| **M2** (~5 wks) | Native TimesFM 2.5; device handling hardened (CPU/CUDA/MPS); `tunable()`/`dials` params; rolling-origin vignette with `tune_grid()` over `rsample::sliding_period()` (plus `modeltime.resample`) | TimesFM parity; CRAN submission |
| **M3** (~5 wks) | Native Chronos-2 (replaces adapter); multivariate targets; conformal vignette with `conformalForecast` | Chronos-2 parity incl. covariate paths |
| **v2** | `tsfm_finetune()` (full + LoRA), AMP where R torch supports it; Moirai-2 | Fine-tuned TTM beats zero-shot on a held-out benchmark |

### 4.6 Risks and mitigations

- **R torch operator gaps** (e.g., fused attention kernels). Mitigation: M1
  model is attention-free; implement attention in composable base ops first,
  optimize later; upstream issues to mlverse early.
- **Upstream model churn** (new checkpoint revisions changing configs).
  Mitigation: revision pinning by default (as brulee does), registry keyed on
  `(architecture, config version)`.
- **Weight licensing.** TTM verified Apache-2.0. Action item: confirm
  Chronos-2 and TimesFM 2.5 weight licenses from their model cards before M2/M3
  ship, and surface the license in `tsfm_capabilities()` (gap item 7 covers
  this by design). The package itself never redistributes weights.
- **CRAN constraints** (no large downloads/tests on CRAN). Mitigation:
  download only on first user call with explicit size message (brulee
  precedent); parity tests run off golden fixtures; `skip_on_cran()` for
  anything touching the Hub.
- **Maintenance burden** is the real long-term risk — this is the reason the
  gap exists. Mitigations: smallest viable catalogue with a third-party
  registration API instead of chasing every model; declarative preprocessing
  specs to keep per-model code thin; propose the package to mlverse or
  tidymodels governance once M2 lands.

### 4.7 What this package deliberately leaves to existing R infrastructure

Backtesting/resampling (`rsample`, `modeltime.resample`), metrics
(`yardstick`, `fabletools::accuracy`), reconciliation
(`fabletools::reconcile`), conformal calibration (`conformalForecast`),
ensembling (`modeltime.ensemble`), deployment (`vetiver`/`plumber`),
hyperparameter search (`tune`/`dials`), and feature engineering (`recipes`).
The integration code the package owns is deliberately thin: the
hardhat/parsnip bridge on the way in, and the `tsfm_forecast` →
fable/modeltime adapters on the way out — confirming the analysis's judgment
that R's surrounding infrastructure is already strong. A further option once
M2 lands is to propose the package (or its architectures) for adoption
alongside `brulee` under tidymodels governance, which would make the
ecosystem fit permanent rather than bolted on.

---

## 5. Sources consulted in the re-run

- brulee (CRAN readme; source `R/chronos2-fit.R`, `R/chronos2-predict.R`):
  <https://cran.r-project.org/package=brulee>, <https://github.com/tidymodels/brulee>
- hfhub: <https://cran.r-project.org/package=hfhub>, <https://github.com/mlverse/hfhub>
- safetensors via Posit AI blog: <https://blogs.rstudio.com/tensorflow/posts/2023-07-12-hugging-face-integrations/>
- skforecast foundation models: <https://skforecast.org/latest/user_guides/foundation-forecasting-models.html>
- Darts FoundationModel (Chronos-2, TimesFM 2.5, TiRex, PatchTST-FM):
  <https://arxiv.org/html/2606.27438v1>, <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html>
- conformalForecast: <https://cran.r-project.org/package=conformalForecast>
- nixtlar: <https://cran.r-project.org/package=nixtlar>
- Model weights: <https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2>,
  <https://huggingface.co/google/timesfm-2.5-200m-pytorch>,
  <https://huggingface.co/amazon/chronos-2>
- Upstream references: <https://github.com/amazon-science/chronos-forecasting>,
  <https://github.com/google-research/timesfm>,
  <https://github.com/ibm-granite/granite-tsfm>