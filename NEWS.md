# tsfm (development version)

First release, covering the model-loading and forecasting foundation plus the
first architectures.

## Core

* `tsfm_pretrained()` loads a model from the Hugging Face Hub (revision pinned)
  through an open architecture registry (`tsfm_register_arch()`).
* `tsfm_capabilities()` reports per-model capability metadata, with pre-flight
  validation that rejects unsupported requests before inference.
* `tsfm_fit()` / `predict()` provide a hardhat-based, tidymodels-conformant
  interface; `forecast()` is a `tsibble`/data-frame convenience path.
* `tsfm_forecast` objects are backed by `distributional`, with `as_fable()` and
  tidy prediction adapters.
* Batched panel inference (`tsfm_run_batches()`) with device resolution and
  validation across CPU/CUDA/MPS (`tsfm_resolve_device()`, `tsfm_set_device()`).

## tidymodels

* `tsfm_reg()` parsnip specification and `"tsfm"` engine: foundation models are
  exchangeable inside a `workflow()` by changing only `set_engine(model_id=)`.
* `context_length()` `dials` parameter for tuning with `tune`/`rsample`.

## Models

* **Chronos-2** (`amazon/chronos-2`): available via a `brulee`-backed adapter.
* **TTM** and **TimesFM 2.5**: registered with capabilities and a documented
  weight-map contract; native numerical ports are in progress, gated on golden
  parity fixtures.
