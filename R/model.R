# A `tsfm_model` is the uniform handle returned by `tsfm_pretrained()`. It wraps
# whatever executes a forward pass (`predict_fn`) together with the metadata the
# rest of the package needs to dispatch, validate, and report. Every
# architecture — the Stage 0 stub, and the native torch modules that follow —
# produces one of these, so the fit/forecast layers never branch on architecture.

#' Construct a model handle
#'
#' @param architecture Character scalar architecture key.
#' @param config Parsed configuration list (from `config.json` or synthesised
#'   for the stub).
#' @param capabilities A [new_tsfm_capabilities()] object.
#' @param predict_fn A function `function(context, h, quantile_levels)` that
#'   forecasts a single numeric series. `context` is the observed history (a
#'   numeric vector, oldest first); it returns a numeric matrix with `h` rows
#'   and `length(quantile_levels)` columns of predictive quantiles.
#' @param model_id,revision Provenance of the checkpoint.
#' @param params Optional opaque parameter/state object (e.g. a torch module);
#'   `NULL` for the weight-free stub.
#' @param predict_batch_fn Optional vectorised forward pass for true batched
#'   inference: `function(contexts, horizons, quantile_levels, device)` where
#'   `contexts`/`horizons` are lists aligned by series, returning a list of
#'   per-series quantile matrices. When `NULL`, [tsfm_run_batches()] falls back
#'   to looping `predict_fn`. Native torch architectures supply this; the stub
#'   does not.
#' @return A `tsfm_model` object.
#' @export
new_tsfm_model <- function(architecture,
                           config,
                           capabilities,
                           predict_fn,
                           model_id = NA_character_,
                           revision = NA_character_,
                           params = NULL,
                           predict_batch_fn = NULL) {
  if (!inherits(capabilities, "tsfm_capabilities")) {
    cli::cli_abort("{.arg capabilities} must be a {.cls tsfm_capabilities} object.")
  }
  if (!is.function(predict_fn)) {
    cli::cli_abort("{.arg predict_fn} must be a function.")
  }
  if (!is.null(predict_batch_fn) && !is.function(predict_batch_fn)) {
    cli::cli_abort("{.arg predict_batch_fn} must be a function or {.code NULL}.")
  }
  structure(
    list(
      architecture     = as.character(architecture),
      config           = config,
      capabilities     = capabilities,
      predict_fn       = predict_fn,
      predict_batch_fn = predict_batch_fn,
      model_id         = as.character(model_id),
      revision         = as.character(revision),
      params           = params
    ),
    class = "tsfm_model"
  )
}

#' @export
tsfm_capabilities.tsfm_model <- function(x, ...) {
  x$capabilities
}

#' @export
print.tsfm_model <- function(x, ...) {
  cli::cli_text("{.cls tsfm_model} {.strong {x$architecture}}")
  cli::cli_text("model_id: {.val {x$model_id}}  revision: {.val {x$revision}}")
  print(x$capabilities)
  invisible(x)
}
