# The forecast object and the two adapter surfaces.
#
# A single inference driver (`tsfm_infer()`) turns per-series histories into a
# `tsfm_forecast`. That object stores, row-aligned:
#   * key + index                (which series, which future step)
#   * .mean                      (point forecast)
#   * .distribution              (a distributional vector, for as_fable())
#   * a quantile matrix + levels (attributes, for the tidy predict() path)
# Both views come from the same predicted quantiles, so the fabletools and
# tidymodels surfaces never have to translate through each other.

# ---- inference driver -------------------------------------------------------

# histories:    named list (by key) of numeric vectors, oldest-first.
# future_index: named list (by key) of future index values, one per horizon.
# The actual forward pass (truncation, batching, device) is delegated to
# tsfm_run_batches(); this driver only assembles the forecast object.
# Returns a `tsfm_forecast`.
tsfm_infer <- function(model, histories, future_index, quantile_levels,
                       key_name = "key", index_name = "index",
                       target = ".response",
                       batch_size = NULL, device = NULL) {
  quantile_levels <- sort(unique(as.numeric(quantile_levels)))
  # Always evaluate the median so the point forecast is exact.
  levels <- sort(unique(c(quantile_levels, 0.5)))
  median_col <- match(0.5, levels)

  keys <- names(histories)
  horizons <- vapply(keys, function(k) length(future_index[[k]]), integer(1))

  qmats <- tsfm_run_batches(model, histories[keys], horizons, levels,
                            batch_size = batch_size, device = device)

  key_out <- vector("list", length(keys))
  idx_out <- vector("list", length(keys))
  qmat_out <- vector("list", length(keys))
  mean_out <- vector("list", length(keys))

  for (i in seq_along(keys)) {
    h <- horizons[[i]]
    if (h == 0L) next
    qmat <- matrix(as.numeric(qmats[[i]]), nrow = h, ncol = length(levels))
    key_out[[i]] <- rep(keys[[i]], h)
    idx_out[[i]] <- future_index[[keys[[i]]]]
    qmat_out[[i]] <- qmat
    mean_out[[i]] <- qmat[, median_col]
  }

  keep <- !vapply(idx_out, is.null, logical(1))
  key_vec <- unlist(key_out[keep], use.names = FALSE)
  idx_vec <- do.call(c, idx_out[keep])
  mean_vec <- unlist(mean_out[keep], use.names = FALSE)
  qmatrix <- do.call(rbind, qmat_out[keep])
  if (is.null(qmatrix)) qmatrix <- matrix(numeric(0), ncol = length(levels))

  # distributional column (only the user-requested levels).
  req_cols <- match(quantile_levels, levels)
  dist <- build_distribution(qmatrix[, req_cols, drop = FALSE], quantile_levels)

  # Build an n-row, 0-column frame first so column assignment sets the rows
  # (assigning a length-n vector into a 0-row frame errors).
  df <- data.frame(matrix(nrow = length(key_vec), ncol = 0))
  df[[key_name]] <- key_vec
  df[[index_name]] <- idx_vec
  df[[".mean"]] <- mean_vec
  df[[".distribution"]] <- dist

  new_tsfm_forecast(
    df,
    key_name = key_name,
    index_name = index_name,
    target = target,
    quantile_levels = quantile_levels,
    quantiles = qmatrix[, req_cols, drop = FALSE],
    levels = quantile_levels
  )
}

# Build a distributional vector of predictive distributions from a quantile
# matrix (rows = observations, cols = `levels`).
build_distribution <- function(qmatrix, levels) {
  rlang::check_installed("distributional", reason = "to store predictive distributions.")
  n <- nrow(qmatrix)
  if (n == 0L) {
    return(distributional::dist_percentile(list(), list()))
  }
  values <- lapply(seq_len(n), function(i) as.numeric(qmatrix[i, ]))
  pcts <- rep(list(as.numeric(levels) * 100), n)
  distributional::dist_percentile(values, pcts)
}

# ---- forecast object --------------------------------------------------------

new_tsfm_forecast <- function(data, key_name, index_name, target,
                              quantile_levels, quantiles, levels) {
  structure(
    data,
    key_name = key_name,
    index_name = index_name,
    target = target,
    quantile_levels = quantile_levels,
    quantiles = quantiles,
    levels = levels,
    class = c("tsfm_forecast", "data.frame")
  )
}

#' @export
print.tsfm_forecast <- function(x, ...) {
  cli::cli_text("{.cls tsfm_forecast} of {.val {attr(x, 'target')}}")
  cli::cli_text(
    "{.val {length(unique(x[[attr(x, 'key_name')]]))}} series x horizon, ",
    "quantile levels {.val {attr(x, 'quantile_levels')}}"
  )
  print(as.data.frame(utils::head(x, 10L)))
  invisible(x)
}

#' @export
as.data.frame.tsfm_forecast <- function(x, ...) {
  attrs <- c("key_name", "index_name", "target", "quantile_levels",
             "quantiles", "levels")
  for (a in attrs) attr(x, a) <- NULL
  class(x) <- "data.frame"
  x
}

# Expand the stored quantile matrix into tidymodels prediction columns:
# .pred (point), .pred_lower / .pred_upper (extreme requested levels), and one
# .pred_qXX column per requested quantile.
tsfm_quantile_columns <- function(fc) {
  levels <- attr(fc, "levels")
  qmatrix <- attr(fc, "quantiles")
  out <- data.frame(.pred = fc[[".mean"]], check.names = FALSE)
  if (length(levels)) {
    out[[".pred_lower"]] <- qmatrix[, which.min(levels)]
    out[[".pred_upper"]] <- qmatrix[, which.max(levels)]
    for (j in seq_along(levels)) {
      nm <- sprintf(".pred_q%02d", round(levels[j] * 100))
      out[[nm]] <- qmatrix[, j]
    }
  }
  out
}

# ---- fable adapter ----------------------------------------------------------

#' Convert a tsfm forecast to a fable
#'
#' Produces a \pkg{fabletools} `fable` so the result flows into
#' `fabletools::accuracy()`, reconciliation, and the tidyverts plotting stack.
#'
#' @param x A `tsfm_forecast`.
#' @param ... Unused.
#' @return A `fable` object.
#' @export
as_fable <- function(x, ...) {
  UseMethod("as_fable")
}

#' @export
as_fable.tsfm_forecast <- function(x, ...) {
  rlang::check_installed(c("tsibble", "fabletools"),
                         reason = "to convert forecasts to a fable.")
  key_name <- attr(x, "key_name")
  index_name <- attr(x, "index_name")
  target <- attr(x, "target")

  df <- as.data.frame(x)
  # Name the distribution column after the response, as fable expects; let
  # fable derive `.mean` from the distribution itself.
  df[[target]] <- df[[".distribution"]]
  df[[".distribution"]] <- NULL
  df[[".mean"]] <- NULL

  # build_tsibble() accepts character key/index, avoiding a tidyselect dep.
  tsbl <- tsibble::build_tsibble(df, key = key_name, index = index_name)
  # `distribution` is captured as a bare symbol by as_fable(); inject it.
  rlang::inject(
    fabletools::as_fable(
      tsbl,
      response = !!target,
      distribution = !!rlang::sym(target)
    )
  )
}

# ---- forecast() convenience path (tsibble / data.frame) ---------------------

#' Forecast future values with a foundation model
#'
#' The convenience surface: hand a panel of history and a horizon, get a
#' `tsfm_forecast` back. The canonical tidymodels path is [tsfm_fit()] +
#' `predict()`; this generic mirrors the tidyverts `forecast()` verb for
#' `tsibble` users.
#'
#' @param object A `tsfm_model`.
#' @param new_data A `tsibble` (index/key inferred) or a `data.frame`.
#' @param h Integer forecast horizon.
#' @param quantile_levels Numeric vector of quantile levels in `(0, 1)`.
#' @param index,key,target Column names; required for the `data.frame` method,
#'   inferred for a `tsibble`.
#' @param batch_size,device Passed to [tsfm_run_batches()].
#' @param ... Unused.
#' @return A `tsfm_forecast`.
#' @export
forecast <- function(object, ...) {
  UseMethod("forecast")
}

#' @rdname forecast
#' @export
forecast.tsfm_model <- function(object, new_data, h = 1L,
                                quantile_levels = c(0.1, 0.5, 0.9),
                                index = NULL, key = NULL, target = NULL,
                                batch_size = NULL, device = NULL, ...) {
  h <- as.integer(h)
  spec <- panel_spec(new_data, index = index, key = key, target = target)

  histories <- split(spec$data[[spec$target]], spec$data[[spec$key]])
  future_index <- future_index_for(spec, h)

  tsfm_infer(
    object,
    histories = histories,
    future_index = future_index,
    quantile_levels = quantile_levels,
    key_name = spec$key,
    index_name = spec$index,
    target = spec$target,
    batch_size = batch_size,
    device = device
  )
}

# Normalise a tsibble or data.frame into a common spec: a plain data.frame plus
# the resolved key/index/target column names. A synthetic single key is added
# when none is supplied so downstream code always groups by a key.
panel_spec <- function(new_data, index = NULL, key = NULL, target = NULL) {
  if (inherits(new_data, "tbl_ts")) {
    rlang::check_installed("tsibble")
    index <- index %||% as.character(tsibble::index_var(new_data))
    key_vars <- tsibble::key_vars(new_data)
    key <- key %||% (if (length(key_vars)) key_vars[[1]] else NULL)
    measured <- setdiff(names(new_data), c(index, key_vars))
    target <- target %||% measured[[1]]
    data <- as.data.frame(new_data)
  } else {
    if (is.null(index) || is.null(target)) {
      cli::cli_abort(
        "For a {.cls data.frame}, both {.arg index} and {.arg target} are required."
      )
    }
    data <- as.data.frame(new_data)
  }
  if (is.null(key)) {
    key <- ".series"
    data[[key]] <- ".series"
  }
  data <- data[order(data[[key]], data[[index]]), , drop = FALSE]
  list(data = data, index = index, key = key, target = target)
}

# Generate `h` future index values per key by extending the observed index by
# its typical step (works for numeric, integer, and Date indices).
future_index_for <- function(spec, h) {
  by_key <- split(spec$data[[spec$index]], spec$data[[spec$key]])
  lapply(by_key, function(idx) extend_index(idx, h))
}

extend_index <- function(idx, h) {
  n <- length(idx)
  if (n == 0L) return(idx[0])
  last <- idx[n]
  step <- if (n >= 2L) stats::median(diff(as.numeric(idx))) else 1
  future_num <- as.numeric(last) + step * seq_len(h)
  if (inherits(idx, "Date")) {
    as.Date(future_num, origin = "1970-01-01")
  } else if (is.integer(idx)) {
    as.integer(round(future_num))
  } else {
    future_num
  }
}
