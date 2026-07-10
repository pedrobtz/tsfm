# TinyTimeMixer (TTM) --- native architecture, Stage 1.
#
# STATUS: scaffold. Capabilities, config parsing, registry dispatch, and the
# weight-map contract are in place; the numerical forward pass and the
# safetensors -> nn_module state-dict mapping land once golden parity fixtures
# are generated. That generation requires resources unavailable in a sandboxed
# CI-only setting (the real ibm-granite/granite-timeseries-ttm-r2 checkpoint
# from the Hub, R torch with a libtorch backend, and the reference `granite-tsfm`
# Python implementation to produce expected outputs). Until then the forward
# pass errors with a clear pointer rather than returning unverified numbers.
#
# Why TTM is the first native model (see roadmap Stage 1): <1M params, a pure
# MLP-mixer with no attention kernels (the lowest R-torch-operator risk),
# Apache-2.0 weights, multivariate + exogenous support, and fast on CPU.
#
# ---- Weight-map contract ----------------------------------------------------
#
# The loader maps safetensors tensor names from the checkpoint to R torch module
# parameters. TTM's published structure (granite-tsfm `TinyTimeMixerForPrediction`)
# is, at a high level:
#
#   backbone.encoder.patcher            patch embedding (Linear over patch_length)
#   backbone.encoder.mixers.{i}.*       N mixer blocks, each:
#       .norm                             LayerNorm / BatchNorm (config: norm_mlp)
#       .mlp_time / .mlp_feature          gated MLP mixing across time / channels
#       .gating_block                     optional gated attention weights
#   decoder.*                           lightweight decoder mixers (mirrors encoder)
#   head.base_forecast_block            Linear projection to prediction_length
#   (optional) head.quantile_*          quantile projection when config has it
#
# The exact tensor names, mixer count, patch/stride, normalisation type, and
# whether channel mixing / gating are enabled are all read from config.json.
# `ttm_weight_map()` below is the single place that translates those names; it
# is intentionally isolated so the port is a matter of filling one function plus
# the nn_module, with the rest of the package (loader, batching, forecast
# object, parsnip) already wired.

ttm_capabilities <- function(config) {
  new_tsfm_capabilities(
    architecture      = "ttm",
    max_context       = as.integer(config$context_length %||% config$seq_len %||% 512L),
    quantiles         = "native",
    multivariate      = TRUE,
    samples           = FALSE,
    past_covariates   = TRUE,
    future_covariates = TRUE,
    static_covariates = FALSE,
    fine_tunable      = TRUE,
    license           = "Apache-2.0"
  )
}

# Translate checkpoint tensor names -> module parameter paths. Returns a named
# character vector (checkpoint name -> module path). Filled during the numerical
# port; kept here so the mapping lives in exactly one place.
ttm_weight_map <- function(config) {
  cli::cli_abort(c(
    "The TTM weight map is not implemented yet.",
    "i" = "It is derived from the checkpoint's {.file config.json} at port time."
  ))
}

# Build the R torch nn_module for TTM from a parsed config. Deferred to the
# numerical port (needs torch); isolated so only this + ttm_weight_map change.
ttm_module <- function(config) {
  rlang::check_installed("torch", reason = "to build the TTM network.")
  cli::cli_abort("The native TTM nn_module is not implemented yet (Stage 1, in progress).")
}

ttm_constructor <- function(config, weights = NULL) {
  caps <- ttm_capabilities(config)

  not_ready <- function(...) {
    cli::cli_abort(c(
      "The native TTM forward pass is not implemented yet (Stage 1, in progress).",
      "i" = "Numerical parity against {.val ibm-granite/granite-timeseries-ttm-r2} \\
             requires golden fixtures generated with the Hub checkpoint, torch, \\
             and reference {.pkg granite-tsfm}.",
      "i" = "In the meantime use {.code tsfm_pretrained(\"stub\")} or the \\
             Chronos-2 adapter via {.code tsfm_pretrained(\"amazon/chronos-2\")}."
    ))
  }

  new_tsfm_model(
    architecture = "ttm",
    config       = config,
    capabilities = caps,
    predict_fn   = not_ready,
    model_id     = config$model_id %||% NA_character_,
    revision     = config$revision %||% NA_character_,
    params       = weights
  )
}
