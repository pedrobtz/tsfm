#' Capability metadata for a time-series foundation model
#'
#' A `tsfm_capabilities` object is a small record describing what a model can
#' and cannot do. It is attached to every model returned by
#' [tsfm_pretrained()] and is used for *pre-flight validation*: a fit or
#' forecast call is rejected with an informative error **before** any tensor
#' work if the request exceeds the model's declared capabilities.
#'
#' @param architecture Character scalar, the architecture key (e.g. `"ttm"`).
#' @param max_context Integer, the maximum supported context length.
#' @param quantiles One of `"native"` (the model emits quantiles directly) or
#'   `"none"`.
#' @param multivariate Logical, whether multiple target series can be modelled
#'   jointly.
#' @param samples Logical, whether the model can emit sample paths.
#' @param past_covariates,future_covariates,static_covariates Logical, which
#'   covariate roles the model consumes.
#' @param fine_tunable Logical, whether the weights can be fine-tuned.
#' @param license Character scalar, the SPDX identifier of the *weight* licence.
#'
#' @return A `tsfm_capabilities` object.
#' @export
new_tsfm_capabilities <- function(architecture,
                                  max_context,
                                  quantiles = c("native", "none"),
                                  multivariate = FALSE,
                                  samples = FALSE,
                                  past_covariates = FALSE,
                                  future_covariates = FALSE,
                                  static_covariates = FALSE,
                                  fine_tunable = FALSE,
                                  license = NA_character_) {
  quantiles <- match.arg(quantiles)
  structure(
    list(
      architecture      = as.character(architecture),
      max_context       = as.integer(max_context),
      quantiles         = quantiles,
      multivariate      = isTRUE(multivariate),
      samples           = isTRUE(samples),
      past_covariates   = isTRUE(past_covariates),
      future_covariates = isTRUE(future_covariates),
      static_covariates = isTRUE(static_covariates),
      fine_tunable      = isTRUE(fine_tunable),
      license           = as.character(license)
    ),
    class = "tsfm_capabilities"
  )
}

#' Report the capabilities of a model
#'
#' @param x A `tsfm_model` (or another object carrying capability metadata).
#' @param ... Unused, for future methods.
#' @return A [new_tsfm_capabilities()] object.
#' @export
tsfm_capabilities <- function(x, ...) {
  UseMethod("tsfm_capabilities")
}

#' @export
tsfm_capabilities.tsfm_capabilities <- function(x, ...) {
  x
}

#' @export
format.tsfm_capabilities <- function(x, ...) {
  yn <- function(v) if (isTRUE(v)) "TRUE" else "FALSE"
  c(
    "<tsfm_capabilities>",
    sprintf("  architecture:      %s", x$architecture),
    sprintf("  max_context:       %s        quantiles: %s", x$max_context, x$quantiles),
    sprintf("  multivariate:      %s        samples:   %s", yn(x$multivariate), yn(x$samples)),
    sprintf("  past_covariates:   %s        future_covariates: %s",
            yn(x$past_covariates), yn(x$future_covariates)),
    sprintf("  static_covariates: %s        fine_tunable: %s",
            yn(x$static_covariates), yn(x$fine_tunable)),
    sprintf("  license:           %s", x$license)
  )
}

#' @export
print.tsfm_capabilities <- function(x, ...) {
  cat(format(x, ...), sep = "\n")
  cat("\n")
  invisible(x)
}

# ---- Pre-flight validation --------------------------------------------------

# Assert a requested context length fits the model. `call` is the calling
# environment for cli's error attribution.
check_context_length <- function(caps, context_length, call = rlang::caller_env()) {
  if (!is.null(context_length) && context_length > caps$max_context) {
    cli::cli_abort(
      c(
        "Requested context length exceeds the model's capability.",
        "x" = "Requested {.val {context_length}} observations of context.",
        "i" = "Architecture {.val {caps$architecture}} supports at most {.val {caps$max_context}}."
      ),
      call = call
    )
  }
  invisible(caps)
}

# Assert the request only asks for capabilities the model declares.
check_capabilities <- function(caps,
                               multivariate = FALSE,
                               past_covariates = FALSE,
                               future_covariates = FALSE,
                               static_covariates = FALSE,
                               call = rlang::caller_env()) {
  problems <- character()
  flag <- function(requested, supported, label) {
    if (isTRUE(requested) && !isTRUE(supported)) {
      problems <<- c(problems, sprintf("%s is not supported by this model.", label))
    }
  }
  flag(multivariate,      caps$multivariate,      "Multivariate targets")
  flag(past_covariates,   caps$past_covariates,   "Past covariates")
  flag(future_covariates, caps$future_covariates, "Future covariates")
  flag(static_covariates, caps$static_covariates, "Static covariates")

  if (length(problems)) {
    cli::cli_abort(
      c(
        "The request exceeds architecture {.val {caps$architecture}}'s capabilities.",
        stats::setNames(problems, rep("x", length(problems)))
      ),
      call = call
    )
  }
  invisible(caps)
}
