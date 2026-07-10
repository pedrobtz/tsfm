test_that("tsfm_reg registers with parsnip and builds a spec", {
  skip_if_not_installed("parsnip")
  make_tsfm_reg()  # idempotent

  expect_true("tsfm_reg" %in% parsnip::get_from_env("models"))

  spec <- tsfm_reg(context_length = 512)
  expect_s3_class(spec, "model_spec")
  expect_identical(spec$mode, "regression")
})

test_that("models are exchangeable by model_id via set_engine only", {
  skip_if_not_installed("parsnip")
  make_tsfm_reg()

  # The Stage 1 exit criterion: the *only* thing that differs between two
  # foundation models in a spec is the engine's model_id argument.
  spec_ttm <- parsnip::set_engine(
    tsfm_reg(), "tsfm",
    model_id = "ibm-granite/granite-timeseries-ttm-r2", index = "date", id = "store"
  )
  spec_chronos <- parsnip::set_engine(
    tsfm_reg(), "tsfm",
    model_id = "amazon/chronos-2", index = "date", id = "store"
  )

  expect_identical(spec_ttm$engine, "tsfm")
  expect_identical(
    rlang::eval_tidy(spec_ttm$eng_args$model_id),
    "ibm-granite/granite-timeseries-ttm-r2"
  )
  expect_identical(
    rlang::eval_tidy(spec_chronos$eng_args$model_id),
    "amazon/chronos-2"
  )
  # Everything else about the two specs is identical.
  expect_identical(names(spec_ttm$eng_args), names(spec_chronos$eng_args))
})

test_that("context_length is exposed as a tunable dials parameter", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("dials")
  skip_if_not_installed("tune")
  make_tsfm_reg()

  param <- context_length()
  expect_s3_class(param, "quant_param")

  tun <- parsnip::tunable(tsfm_reg(context_length = tune::tune()))
  expect_true("context_length" %in% tun$name)
})
