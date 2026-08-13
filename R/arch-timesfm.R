# TimesFM 2.5 --- 0.1.0 release-model scaffold.
#
# STATUS: scaffold, mirroring the TTM approach (see arch-ttm.R). Capabilities,
# config parsing, registry dispatch, and the weight-map contract are wired; the
# numerical forward pass and the safetensors -> nn_module mapping land once
# golden parity fixtures are generated against google/timesfm-2.5-200m-pytorch
# (needs the Hub checkpoint, R torch with a libtorch backend, and the reference
# `timesfm` Python package). Until then the forward pass errors with a clear
# pointer rather than returning unverified numbers.
#
# Why TimesFM is the release model (see roadmap): highest-demand missing model; a
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
    max_horizon       = as.integer(
      config$quantile_horizon_length %||% config$quantile_horizon_len %||% 1024L
    ),
    quantiles         = "native",
    quantile_levels   = as.numeric(config$quantiles %||% seq(0.1, 0.9, by = 0.1)),
    multivariate      = FALSE,
    samples           = FALSE,
    past_covariates   = FALSE,
    future_covariates = FALSE,
    static_covariates = FALSE,
    fine_tunable      = FALSE,
    license           = "Apache-2.0"
  )
}

# Translate checkpoint tensor names -> module parameter paths. Filled during the
# numerical port; isolated so the mapping lives in exactly one place.
timesfm_weight_map <- function(config) {
  tsfm_abort_checkpoint(c(
    "The TimesFM weight map is not implemented yet.",
    "i" = "It is derived from the checkpoint's {.file config.json} at port time."
  ),
  model_id = config$model_id %||% "google/timesfm-2.5-200m-pytorch",
  revision = config$revision %||% NA_character_,
  expected = "complete state-dict mapping",
  actual = "scaffold")
}

# Build the R torch nn_module for TimesFM. Deferred to the numerical port; the
# causal attention block is the main new operator surface versus TTM.
timesfm_module <- function(config) {
  tsfm_require_namespace("torch", reason = "It is needed to build the TimesFM network.")
  tsfm_abort_checkpoint(
    "The native TimesFM nn_module is not implemented yet.",
    model_id = config$model_id %||% "google/timesfm-2.5-200m-pytorch",
    revision = config$revision %||% NA_character_,
    expected = "checkpoint-compatible nn_module",
    actual = "scaffold"
  )
}

timesfm_constructor <- function(config, weights = NULL) {
  caps <- timesfm_capabilities(config)

  not_ready <- function(...) {
    tsfm_abort_capability(c(
      "The native TimesFM forward pass is not implemented yet.",
      "i" = "Numerical parity against {.val google/timesfm-2.5-200m-pytorch} \\
             requires golden fixtures generated with the Hub checkpoint, torch, \\
             and reference {.pkg timesfm}.",
      "i" = "Use {.code tsfm_pretrained(\"stub\")} to exercise the engine shell."
    ),
    model_id = config$model_id %||% "google/timesfm-2.5-200m-pytorch",
    revision = config$revision %||% NA_character_,
    capability = "model_state",
    requested = "scaffold",
    supported = "supported")
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
