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

# Validate a device string. Accepts "auto", "cpu", "mps", "cuda", and indexed
# CUDA devices like "cuda:1".
is_valid_device <- function(device) {
  length(device) == 1L && is.character(device) &&
    grepl("^(auto|cpu|mps|cuda(:[0-9]+)?)$", device)
}

#' Resolve a compute device
#'
#' Returns a concrete device string for torch models. `"auto"` (the default,
#' overridable with `options(tsfm.device = ...)`) picks CUDA, then MPS, then CPU,
#' based on what the installed torch backend reports, and warns if a requested
#' accelerator is unavailable, falling back to CPU. Without torch it is always
#' `"cpu"`, so the stub path never touches torch.
#'
#' @param device One of `"auto"`, `"cpu"`, `"mps"`, `"cuda"`, `"cuda:N"`, or
#'   `NULL` to read the `tsfm.device` option.
#' @return A concrete device string (never `"auto"`).
#' @export
tsfm_resolve_device <- function(device = NULL) {
  device <- device %||% getOption("tsfm.device", "auto")
  if (!is_valid_device(device)) {
    cli::cli_abort(c(
      "Invalid {.arg device}: {.val {device}}.",
      "i" = "Use one of {.val auto}, {.val cpu}, {.val mps}, {.val cuda}, or {.val cuda:N}."
    ))
  }
  if (!identical(device, "auto")) {
    return(validate_available_device(device))
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

# For an explicitly requested accelerator, confirm the backend actually has it;
# warn and fall back to CPU rather than erroring deep in a forward pass.
validate_available_device <- function(device) {
  backend <- sub(":.*$", "", device)
  if (backend == "cpu" || !requireNamespace("torch", quietly = TRUE)) {
    return(device)
  }
  available <- switch(
    backend,
    cuda = isTRUE(tryCatch(torch::cuda_is_available(), error = function(e) FALSE)),
    mps  = isTRUE(tryCatch(torch::backends_mps_is_available(), error = function(e) FALSE)),
    TRUE
  )
  if (!available) {
    cli::cli_warn(c(
      "Requested device {.val {device}} is not available; falling back to {.val cpu}.",
      "i" = "Check your torch installation and drivers."
    ))
    return("cpu")
  }
  device
}

#' Set or get the default tsfm compute device
#'
#' Thin wrapper over the `tsfm.device` option consulted by
#' [tsfm_resolve_device()].
#'
#' @param device A device string, or `NULL` to only read the current setting.
#' @return Invisibly, the previous option value.
#' @export
tsfm_set_device <- function(device) {
  if (!is_valid_device(device)) {
    cli::cli_abort("Invalid {.arg device}: {.val {device}}.")
  }
  old <- getOption("tsfm.device", "auto")
  options(tsfm.device = device)
  invisible(old)
}

# Move a torch tensor or module to a device. No-op fallback keeps non-torch
# code paths (the stub) working; native architectures call this in their
# forward pass.
tsfm_to_device <- function(x, device) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    return(x)
  }
  x$to(device = torch::torch_device(device))
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
