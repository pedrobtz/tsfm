test_that("all built-in architectures are registered", {
  expect_setequal(tsfm_registry_archs(), c("stub", "ttm", "timesfm"))
  expect_false(tsfm_registry_has("chronos2"))
})

test_that("architecture keys are normalised from Hub config conventions", {
  expect_identical(normalize_architecture(list(model_type = "TinyTimeMixer")), "ttm")
  expect_identical(
    normalize_architecture(list(architectures = "TinyTimeMixerForPrediction")),
    "ttm"
  )
  expect_identical(normalize_architecture(list(model_type = "TimesFM")), "timesfm")
  expect_identical(
    normalize_architecture(list(architectures = "PatchedTimeSeriesDecoder")),
    "timesfm"
  )
  expect_identical(normalize_architecture(list(model_type = "ChronosBolt")), "chronos2")
  expect_identical(normalize_architecture(list(model_type = "PatchTST")), "patchtst")
})

test_that("Chronos-2 is rejected before network or adapter work", {
  expect_true(is_chronos2_id("amazon/chronos-2"))
  expect_true(is_chronos2_id("amazon/chronos2"))
  expect_false(is_chronos2_id("google/timesfm-2.5-200m-pytorch"))

  expect_error(
    tsfm_pretrained("amazon/chronos-2"),
    "not a supported model in tsfm 0.1.0",
    fixed = TRUE
  )
})

test_that("TTM scaffold advertises capabilities but defers the forward pass", {
  model <- ttm_constructor(list(context_length = 1536L))
  expect_identical(model$architecture, "ttm")
  caps <- tsfm_capabilities(model)
  expect_identical(caps$max_context, 1536L)
  expect_identical(caps$quantiles, "none")
  expect_false(caps$multivariate)
  expect_false(caps$past_covariates)
  expect_false(caps$future_covariates)
  expect_false(caps$fine_tunable)
  expect_identical(caps$license, "Apache-2.0")
  # Numerical port not done: forward pass must error, not fabricate output.
  expect_error(model$predict_fn(1:10, 3, c(0.1, 0.5, 0.9)), "not implemented")
})

test_that("TimesFM scaffold advertises capabilities but defers the forward pass", {
  model <- timesfm_constructor(list(context_length = 2048L))
  expect_identical(model$architecture, "timesfm")
  caps <- tsfm_capabilities(model)
  expect_identical(caps$max_context, 2048L)
  expect_false(caps$multivariate)
  expect_false(caps$past_covariates)
  expect_false(caps$future_covariates)
  expect_false(caps$fine_tunable)
  expect_identical(caps$quantiles, "native")
  expect_identical(caps$license, "Apache-2.0")
  expect_error(model$predict_fn(1:10, 3, c(0.1, 0.5, 0.9)), "not implemented")
})
