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
