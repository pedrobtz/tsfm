# Batched panel inference and device placement.
#
# This module is the single choke point through which every architecture is
# executed, so context truncation, batching, and device selection live in one
# place rather than being re-implemented per model. The stub runs series-by-
# series via `predict_fn`; native torch architectures supply a vectorised
# `predict_batch_fn` and are fed in chunks of `batch_size`.

# Default batch size, overridable with `options(tsfm.batch_size = ...)`.
tsfm_default_batch_size <- function() {
  as.integer(getOption("tsfm.batch_size", 64L))
}

#' Resolve a compute device
#'
#' Returns a device string for torch models. `"auto"` (the default, overridable
#' with `options(tsfm.device = ...)`) picks CUDA, then MPS, then CPU, based on
#' what the installed torch backend reports. Without torch it is always
#' `"cpu"`, so the stub path never touches torch.
#'
#' @param device One of `"auto"`, `"cpu"`, `"cuda"`, `"mps"`, or `NULL` to read
#'   the option.
#' @return A device string.
#' @export
tsfm_resolve_device <- function(device = NULL) {
  device <- device %||% getOption("tsfm.device", "auto")
  if (!identical(device, "auto")) {
    return(device)
  }
  if (!requireNamespace("torch", quietly = TRUE)) {
    return("cpu")
  }
  if (isTRUE(tryCatch(torch::cuda_is_available(), error = function(e) FALSE))) {
    return("cuda")
  }
  if (isTRUE(tryCatch(torch::backends_mps_is_available(), error = function(e) FALSE))) {
    return("mps")
  }
  "cpu"
}

#' Run a model over many series in batches
#'
#' Truncates each context to the model's `max_context`, then evaluates the
#' model. Returns a list, aligned to `contexts`, of `h x length(quantile_levels)`
#' predictive-quantile matrices.
#'
#' @param model A `tsfm_model`.
#' @param contexts A list of numeric context vectors (oldest first).
#' @param horizons A list/vector of per-series integer horizons.
#' @param quantile_levels Numeric vector of quantile levels.
#' @param batch_size Series per batch for vectorised models; default from
#'   [tsfm_default_batch_size()].
#' @param device Device string; resolved via [tsfm_resolve_device()].
#' @return A list of quantile matrices.
#' @export
tsfm_run_batches <- function(model, contexts, horizons, quantile_levels,
                             batch_size = NULL, device = NULL) {
  caps <- model$capabilities
  contexts <- lapply(contexts, function(ctx) {
    ctx <- as.numeric(ctx)
    ctx <- ctx[!is.na(ctx)]
    if (length(ctx) > caps$max_context) utils::tail(ctx, caps$max_context) else ctx
  })
  horizons <- as.integer(unlist(horizons, use.names = FALSE))
  n <- length(contexts)
  if (n == 0L) {
    return(list())
  }
  batch_size <- as.integer(batch_size %||% tsfm_default_batch_size())
  device <- tsfm_resolve_device(device)

  if (is.function(model$predict_batch_fn)) {
    out <- vector("list", n)
    groups <- split(seq_len(n), (seq_len(n) - 1L) %/% batch_size)
    for (g in groups) {
      res <- model$predict_batch_fn(contexts[g], horizons[g], quantile_levels,
                                    device = device)
      if (length(res) != length(g)) {
        cli::cli_abort(
          "The model's batch function returned {length(res)} results for a \\
           batch of {length(g)} series."
        )
      }
      out[g] <- res
    }
    out
  } else {
    lapply(seq_len(n), function(i) {
      model$predict_fn(contexts[[i]], horizons[[i]], quantile_levels)
    })
  }
}
