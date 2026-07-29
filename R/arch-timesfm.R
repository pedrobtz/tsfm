# TimesFM 2.5 --- native architecture, Stage 2.
#
# STATUS: scaffold, mirroring the TTM approach (see arch-ttm.R). Capabilities,
# config parsing, registry dispatch, and the weight-map contract are wired; the
# numerical forward pass and the safetensors -> nn_module mapping land once
# golden parity fixtures are generated against google/timesfm-2.5-200m-pytorch
# (needs the Hub checkpoint, R torch with a libtorch backend, and the reference
# `timesfm` Python package). Until then the forward pass errors with a clear
# pointer rather than returning unverified numbers.
#
# Why TimesFM is Stage 2 (see roadmap): highest-demand missing model; a
# decoder-only patching transformer, so it is the first architecture to exercise
# real attention blocks, quantile heads, and long contexts in R torch.
#
# ---- Weight-map contract ----------------------------------------------------
#
# TimesFM 2.5 (decoder-only patched transformer, per google-research/timesfm):
#
#   input_ff_layer / residual_block   patch embedding: reshape context into
#                                     patches of `patch_len`, project to d_model
#   stacked_transformer.layers.{i}.*  N decoder layers, each:
#       .self_attn                      causal multi-head attention over patches
#       .mlp                            position-wise feed-forward
#       .*_layernorm                    pre/post norms
#   freq_emb                          frequency embedding (per-series period)
#   horizon_ff_layer / output head    project final patch -> output_patch_len x
#                                     n_quantiles (native quantile forecasts)
#
# patch_len, context_len, horizon_len, model_dim, num_layers, num_heads, and the
# quantile list are all read from config.json. `timesfm_weight_map()` is the one
# place that translates checkpoint tensor names; `timesfm_module()` builds the
# nn_module. Everything else (loader, batching, forecast object, parsnip) is
# already wired, so the port is those two functions plus the attention block.

timesfm_capabilities <- function(config) {
  new_tsfm_capabilities(
    architecture      = "timesfm",
    max_context       = as.integer(
      config$context_length %||% config$context_len %||% config$max_context %||% 2048L
    ),
    quantiles         = "native",
    multivariate      = FALSE,   # univariate targets; covariates via regression
    samples           = FALSE,
    past_covariates   = TRUE,
    future_covariates = TRUE,
    static_covariates = FALSE,
    fine_tunable      = TRUE,
    license           = "Apache-2.0"
  )
}

# Translate checkpoint tensor names -> module parameter paths. Filled during the
# numerical port; isolated so the mapping lives in exactly one place.
timesfm_weight_map <- function(config) {
  cli::cli_abort(c(
    "The TimesFM weight map is not implemented yet.",
    "i" = "It is derived from the checkpoint's {.file config.json} at port time."
  ))
}

# Build the R torch nn_module for TimesFM. Deferred to the numerical port; the
# causal attention block is the main new operator surface versus TTM.
timesfm_module <- function(config) {
  rlang::check_installed("torch", reason = "to build the TimesFM network.")
  cli::cli_abort("The native TimesFM nn_module is not implemented yet (Stage 2, in progress).")
}

timesfm_constructor <- function(config, weights = NULL) {
  caps <- timesfm_capabilities(config)

  not_ready <- function(...) {
    cli::cli_abort(c(
      "The native TimesFM forward pass is not implemented yet (Stage 2, in progress).",
      "i" = "Numerical parity against {.val google/timesfm-2.5-200m-pytorch} \\
             requires golden fixtures generated with the Hub checkpoint, torch, \\
             and reference {.pkg timesfm}.",
      "i" = "In the meantime use {.code tsfm_pretrained(\"stub\")} or the \\
             Chronos-2 adapter via {.code tsfm_pretrained(\"amazon/chronos-2\")}."
    ))
  }

  new_tsfm_model(
    architecture = "timesfm",
    config       = config,
    capabilities = caps,
    predict_fn   = not_ready,
    model_id     = config$model_id %||% NA_character_,
    revision     = config$revision %||% NA_character_,
    params       = weights
  )
}
