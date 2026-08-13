test_that("capabilities constructor coerces and defaults", {
  caps <- new_tsfm_capabilities(
    "timesfm",
    max_context = 1536,
    max_horizon = 1024,
    quantiles = "native",
    quantile_levels = c(0.9, 0.1, 0.5),
    license = "Apache-2.0"
  )
  expect_s3_class(caps, "tsfm_capabilities")
  expect_identical(caps$max_context, 1536L)
  expect_identical(caps$max_horizon, 1024)
  expect_equal(caps$quantile_levels, c(0.1, 0.5, 0.9))
  expect_false(caps$multivariate)
  expect_false(caps$samples)
  expect_identical(caps$license, "Apache-2.0")
})

test_that("contract v1 refuses declarations with no execution channel", {
  error <- expect_error(
    new_tsfm_capabilities("bad", 128L, multivariate = TRUE),
    class = "tsfm_error_contract"
  )
  expect_s3_class(error, "tsfm_error_internal")
  expect_identical(error$contract, "capability declaration")
})

test_that("capabilities print is informative", {
  caps <- new_tsfm_capabilities("stub", max_context = 512)
  out <- capture.output(print(caps))
  expect_true(any(grepl("architecture:", out)))
  expect_true(any(grepl("license:", out)))
})

test_that("pre-flight rejects requests beyond capability", {
  caps <- new_tsfm_capabilities("stub", max_context = 100, multivariate = FALSE)
  expect_error(check_context_length(caps, 200), "context length")
  expect_error(check_capabilities(caps, multivariate = TRUE), "Multivariate")
  expect_error(check_capabilities(caps, future_covariates = TRUE), "covariates")
  expect_silent(check_capabilities(caps, multivariate = FALSE))
  expect_silent(check_context_length(caps, 100))
})

test_that("explicit quantile levels and horizon are enforced", {
  caps <- new_tsfm_capabilities(
    "timesfm", 16384L,
    max_horizon = 1024L,
    quantile_levels = c(0.1, 0.5, 0.9)
  )
  expect_equal(check_quantile_levels(caps, c(0.9, 0.1)), c(0.1, 0.9))
  expect_error(
    check_quantile_levels(caps, c(0.05, 0.5)),
    class = "tsfm_error_quantile_levels"
  )
  expect_error(check_horizon(caps, 0), class = "tsfm_error_capability")
  expect_error(check_horizon(caps, 1025), class = "tsfm_error_capability")
  expect_silent(check_horizon(caps, 1024))
})

# seq(0.1, 0.9, by = 0.1) accumulates rounding error, so its 3rd and 7th values
# are not the doubles a JSON config parses for 0.3 and 0.7. Both spellings name
# the same trained levels, and the catalogue and config.json each use a
# different one, so the engine must reconcile them rather than compare bits.
test_that("the two spellings of the trained levels really do differ", {
  literal <- jsonlite::fromJSON("[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]")
  generated <- seq(0.1, 0.9, by = 0.1)
  expect_false(identical(literal, generated))
  expect_equal(literal, generated)
  expect_identical(which(literal != generated), c(3L, 7L))
})

test_that("requested quantile levels are matched tolerantly and canonicalised", {
  literal <- jsonlite::fromJSON("[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]")
  caps <- new_tsfm_capabilities("timesfm", 16384L, quantile_levels = literal)

  resolved <- check_quantile_levels(caps, seq(0.1, 0.9, by = 0.1))
  # The checkpoint's own values come back, not the caller's spelling: an
  # architecture selecting an output channel by position must not have to
  # re-derive a match the engine already made.
  expect_identical(resolved, literal)

  expect_identical(tsfm_match_quantile_levels(seq(0.1, 0.9, by = 0.1), literal), 1:9)
  expect_identical(tsfm_match_quantile_levels(c(0.05, 0.5), literal), c(NA_integer_, 5L))
})

test_that("levels collapsing onto one trained level are rejected", {
  literal <- jsonlite::fromJSON("[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]")
  caps <- new_tsfm_capabilities("timesfm", 16384L, quantile_levels = literal)
  error <- expect_error(
    check_quantile_levels(caps, c(0.3, 0.30000000000000004, 0.5)),
    class = "tsfm_error_quantile_levels"
  )
  expect_s3_class(error, "tsfm_error_recoverable")
})
