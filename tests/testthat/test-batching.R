test_that("batching loops predict_fn, aligned to inputs, when no batch fn", {
  set.seed(1)
  model <- tsfm_pretrained("stub")
  contexts <- list(a = cumsum(rnorm(30)) + 100, b = cumsum(rnorm(40)) + 50)
  res <- tsfm_run_batches(model, contexts, c(3, 5), c(0.1, 0.5, 0.9))

  expect_length(res, 2)
  expect_equal(dim(res[[1]]), c(3L, 3L))
  expect_equal(dim(res[[2]]), c(5L, 3L))
  expect_equal(res[[1]][, 2], rep(contexts$a[30], 3))  # median == last value
  expect_equal(res[[2]][, 2], rep(contexts$b[40], 5))
})

test_that("batching truncates context to max_context", {
  model <- tsfm_pretrained("stub")
  model$capabilities$max_context <- 10L
  res <- tsfm_run_batches(model, list(x = 1:100), 2, c(0.1, 0.5, 0.9))
  expect_equal(res[[1]][, 2], rep(100, 2))  # kept the most recent value
})

test_that("a vectorised predict_batch_fn is chunked by batch_size", {
  set.seed(2)
  base <- tsfm_pretrained("stub")
  seen <- integer(0)
  batch <- function(contexts, horizons, quantile_levels, device) {
    seen <<- c(seen, length(contexts))
    lapply(seq_along(contexts), function(i) {
      base$predict_fn(contexts[[i]], horizons[[i]], quantile_levels)
    })
  }
  base$predict_batch_fn <- batch

  contexts <- stats::setNames(lapply(1:5, function(i) rnorm(20)), letters[1:5])
  res <- tsfm_run_batches(base, contexts, rep(2, 5), c(0.25, 0.75), batch_size = 2)
  expect_length(res, 5)
  expect_equal(seen, c(2L, 2L, 1L))  # 5 series in batches of 2
})

test_that("engine-boundary request validation happens before inference", {
  model <- tsfm_pretrained("stub")
  model$predict_fn <- function(...) stop("inference was reached")

  expect_error(
    tsfm_run_batches(model, list(1:3), 0, 0.5),
    class = "tsfm_error_capability"
  )
  expect_error(
    tsfm_run_batches(model, list(1:3), 1.5, 0.5),
    class = "tsfm_error_capability"
  )
  expect_error(
    tsfm_run_batches(model, list(1:3), 1, 0),
    class = "tsfm_error_quantile_levels"
  )
  expect_error(
    tsfm_run_batches(model, list(numeric()), 1, 0.5),
    class = "tsfm_error_capability"
  )
  expect_error(
    tsfm_run_batches(model, list(1:3), 1, 0.5, batch_size = 0),
    class = "tsfm_error_capability"
  )
})

test_that("architecture return matrices are validated at the engine boundary", {
  base <- tsfm_pretrained("stub")
  base$predict_fn <- function(context, h, quantile_levels) {
    matrix(0, nrow = h + 1L, ncol = length(quantile_levels))
  }
  error <- expect_error(
    tsfm_run_batches(base, list(1:3), 2, c(0.1, 0.5)),
    class = "tsfm_error_contract"
  )
  expect_identical(error$contract, "predict return shape")
  expect_identical(error$expected, c(2L, 2L))
})

test_that("unsupported trained quantiles fail before a scaffold forward pass", {
  model <- timesfm_constructor(list(
    context_length = 16384L,
    quantile_horizon_length = 1024L,
    quantiles = seq(0.1, 0.9, by = 0.1)
  ))
  error <- expect_error(
    tsfm_run_batches(model, list(1:32), 2, c(0.05, 0.5)),
    class = "tsfm_error_quantile_levels"
  )
  expect_equal(error$requested, c(0.05, 0.5))
  expect_equal(error$supported, seq(0.1, 0.9, by = 0.1))
})
