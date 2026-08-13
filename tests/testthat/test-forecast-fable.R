# Stage 0 exit gate (2/2): forecast() |> as_fable() |> fabletools::accuracy().

test_that("forecast() produces a tsfm_forecast from a data.frame", {
  skip_if_not_installed("distributional")

  set.seed(1)
  df <- data.frame(t = 1:30, y = cumsum(rnorm(30)) + 100)
  model <- tsfm_pretrained("stub")
  fc <- forecast(model, df, h = 6, index = "t", target = "y")

  expect_s3_class(fc, "tsfm_forecast")
  expect_identical(nrow(fc), 6L)
  expect_true(".distribution" %in% names(fc))
  expect_equal(fc$t, 31:36)
})

test_that("forecast() |> as_fable() |> accuracy() works on the stub", {
  skip_if_not_installed("distributional")
  skip_if_not_installed("tsibble")
  skip_if_not_installed("fabletools")

  set.seed(3)
  full <- rbind(
    data.frame(store = "a", t = 1:36, y = cumsum(rnorm(36)) + 100),
    data.frame(store = "b", t = 1:36, y = cumsum(rnorm(36)) + 50)
  )
  train <- full[full$t <= 30, ]

  tsb_train <- tsibble::as_tsibble(train, key = store, index = t)
  tsb_full <- tsibble::as_tsibble(full, key = store, index = t)

  model <- tsfm_pretrained("stub")
  fc <- forecast(model, tsb_train, h = 6, quantile_levels = c(0.1, 0.5, 0.9))
  expect_s3_class(fc, "tsfm_forecast")

  # Qualified: tsfm registers a method on fabletools' generic rather than
  # exporting a competing `as_fable()`.
  fbl <- fabletools::as_fable(fc)
  expect_true(inherits(fbl, "fbl_ts"))

  acc <- fabletools::accuracy(fbl, tsb_full)
  expect_true(all(is.finite(acc$RMSE)))
})
