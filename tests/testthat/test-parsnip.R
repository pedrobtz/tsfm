test_that("tsfm_reg registers with parsnip and builds a spec", {
  skip_if_not_installed("parsnip")
  make_tsfm_reg()  # idempotent

  expect_true("tsfm_reg" %in% parsnip::get_from_env("models"))

  spec <- tsfm_reg(context_length = 512)
  expect_s3_class(spec, "model_spec")
  expect_identical(spec$mode, "regression")
})

test_that("model identity is supplied through set_engine", {
  skip_if_not_installed("parsnip")
  make_tsfm_reg()

  spec_stub <- parsnip::set_engine(
    tsfm_reg(), "tsfm",
    model_id = "stub", index = "date", id = "store"
  )
  spec_other <- parsnip::set_engine(
    tsfm_reg(), "tsfm",
    model_id = "stub-alternate", index = "date", id = "store"
  )

  expect_identical(spec_stub$engine, "tsfm")
  expect_identical(
    rlang::eval_tidy(spec_stub$eng_args$model_id),
    "stub"
  )
  expect_identical(
    rlang::eval_tidy(spec_other$eng_args$model_id),
    "stub-alternate"
  )
  expect_identical(names(spec_stub$eng_args), names(spec_other$eng_args))
})

test_that("parsnip fits and predicts through the exported stub bridge", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("hardhat")
  skip_if_not_installed("distributional")
  make_tsfm_reg()

  expect_true("tsfm_parsnip_fit" %in% getNamespaceExports("tsfm"))

  train <- data.frame(store = "a", date = 1:20, value = as.numeric(1:20))
  future <- data.frame(store = "a", date = 21:23)
  spec <- parsnip::set_engine(
    tsfm_reg(context_length = 8L),
    "tsfm",
    model_id = "stub",
    index = "date",
    id = "store"
  )

  fit <- parsnip::fit(spec, value ~ 1, data = train)
  expect_s3_class(fit$fit, "tsfm_fit")
  expect_identical(fit$fit$model$capabilities$max_context, 8L)

  pred <- predict(fit, new_data = future)
  expect_identical(names(pred), ".pred")
  expect_equal(pred$.pred, rep(20, 3))
})

test_that("context_length is exposed as a tunable dials parameter", {
  skip_if_not_installed("dials")

  param <- context_length()
  expect_s3_class(param, "quant_param")

  # Exercise the tunable metadata directly (the generic's home package varies
  # across tidymodels versions; dispatch is wired in make_tsfm_reg()).
  tun <- tunable_tsfm_reg(tsfm_reg())
  expect_true("context_length" %in% tun$name)
  expect_identical(tun$call_info[[1]]$fun, "context_length")
})
