# tsfm 0.0.0.9000

Development release establishing the model-loading and forecasting shell. No
real foundation model is supported yet.

## Engine contract

* `tsfm` is positioned as a framework-neutral **inference engine**: it owns the
  path from a pretrained checkpoint to predictive quantiles, and interfaces to
  tidymodels, the tidyverts, and modeltime are optional adapters of equal
  standing. `hardhat` moved from `Imports` to `Suggests` accordingly.
* `` ?`tsfm-architecture-contract` `` documents the public plugin surface, and
  `tsfm_contract_version()` versions it. `new_tsfm_model()` stamps every handle.
* `tsfm_check_architecture()` turns the contract into an executable gate:
  construction, forecast shape, quantile monotonicity, finiteness, context
  limits, empty-context handling, and batch/loop agreement. Built-in
  architectures run it in the test suite; third parties get the same check
  without reading the engine's internals.
* `forecast()` is now re-exported from `generics` rather than defined locally,
  and `as_fable()` is registered onto `fabletools`'s generic from `.onLoad()`
  instead of shadowing it. Attaching `tsfm` alongside fabletools, modeltime, or
  forecast no longer risks masking either verb. Call `fabletools::as_fable()`.
* Fixed a latent bug where the parsnip `tunable()` registration called a
  non-existent `rlang::s3_register()`; it is `vctrs::s3_register()`, and the
  failure was silently swallowed by `.onLoad()`'s `tryCatch`, so the engine's
  tuning support never actually registered. Failures now warn.

## Core

* Added `tsfm_download()` for explicit manifest prefetch, `tsfm_cache_status()`
  for separate disk/resident state, and `tsfm_unload()` for resident eviction
  without deleting the `hfhub` cache.
* `tsfm_pretrained()` now keys constructed handles by checkpoint, immutable
  revision, resolved device, and load-affecting options in an R-session LRU
  bounded by `options(tsfm.max_loaded_models = 1L)`. `reuse = FALSE` bypasses
  reuse and `0L` disables resident storage.
* Added safetensors header validation and named R `torch` state-dict loading,
  with download and checkpoint failures mapped to their structured policy
  families.
* Added the offline `tsfm_models()` checkpoint catalogue. Its safe default
  returns only supported checkpoints, while `state = NULL` exposes pinned
  scaffold records and static cost, capability, manifest-cache, provenance,
  and weight-licence metadata without network access.
* Added a structured condition hierarchy rooted at `tsfm_error`, with one
  recoverable, external, or internal policy parent and structured fields on
  capability, context, quantile, device, download, checkpoint, and contract
  failures.
* Contract v1 now records explicit trained quantile levels and a maximum
  horizon, rejects unsupported probabilities before inference, and refuses
  capability declarations for multivariate/covariate/sample/fine-tuning
  channels that do not yet exist.
* The batch boundary validates contexts, horizons, quantile levels, batch size,
  and architecture return shape/finiteness/monotonicity before assembling a
  forecast.
* `tsfm_pretrained()` loads the self-contained `stub` fixture and provides the
  Hub-resolution shell for future supported checkpoints through an open
  architecture registry (`tsfm_register_arch()`).
* `tsfm_capabilities()` reports per-model capability metadata, with pre-flight
  validation that rejects unsupported requests before inference.
* `forecast()` forecasts a `tsibble` or data-frame panel with no optional
  dependencies; `tsfm_fit()` / `predict()` provide a hardhat-based,
  tidymodels-conformant interface when `hardhat` is installed.
* `tsfm_forecast` objects are backed by `distributional`, with a
  `fabletools::as_fable()` method and tidy prediction adapters.
* Batched panel inference (`tsfm_run_batches()`) with device resolution and
  validation across CPU/CUDA/MPS (`tsfm_resolve_device()`, `tsfm_set_device()`).

## tidymodels

* `tsfm_reg()` parsnip specification and `"tsfm"` engine, with an exported fit
  bridge exercised end to end against the stub.
* `context_length()` is a `dials` parameter and now actually bounds the history
  retained by the fitted engine.

## Models

* Added four compact, committed TimesFM reference fixtures from the pinned
  official implementation: typical, short-context, context-truncation, and
  two-series batch/loop cases. The locked generator records every forecast
  flag and upstream dependency pin.
* Captured and validated the exact pinned TimesFM state layout: 232 float32
  tensors and 231,289,280 parameters. The full 925,181,104-byte checkpoint
  loads into a named R state dict; native module construction remains Stage 3.
* The native TimesFM feasibility gate passed: a reduced R `torch` transformer
  block and continuous quantile head match the pinned official PyTorch output
  below `1e-6`; the exact-width CPU spike executes successfully. This does not
  yet constitute checkpoint support.
* **Stub** is the only executable built-in and is explicitly a random-walk test
  fixture, not a foundation model.
* **TimesFM 2.5** is the `0.1.0` release-model scaffold; it is not supported
  until conformance and numerical-parity gates pass.
* **TTM** remains a registered scaffold deferred until the engine represents
  point-only output.
* **Chronos-2** is no longer registered or advertised as available. Its
  unverified Brulee adapter remains reference work only.
