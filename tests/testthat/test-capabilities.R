test_that("capabilities constructor coerces and defaults", {
  caps <- new_tsfm_capabilities("ttm", max_context = 1536, quantiles = "native",
                                multivariate = TRUE, license = "Apache-2.0")
  expect_s3_class(caps, "tsfm_capabilities")
  expect_identical(caps$max_context, 1536L)
  expect_true(caps$multivariate)
  expect_false(caps$samples)
  expect_identical(caps$license, "Apache-2.0")
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
