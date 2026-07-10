# Stage 0 exit gate (1/2): fit -> predict -> yardstick::rmse().

test_that("fit -> predict -> yardstick::rmse works on the stub", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("distributional")
  skip_if_not_installed("yardstick")

  set.seed(1)
  train <- data.frame(store = "a", t = 1:40, y = cumsum(rnorm(40)) + 100)
  future <- data.frame(store = "a", t = 41:45, y = cumsum(rnorm(5)) + 100)

  model <- tsfm_pretrained("stub")
  fit <- tsfm_fit(y ~ 1, data = train, model = model, index = "t", id = "store")
  expect_s3_class(fit, "tsfm_fit")

  preds <- predict(fit, new_data = future)
  expect_identical(nrow(preds), 5L)
  expect_true(all(c(".pred", ".pred_lower", ".pred_upper") %in% names(preds)))
  expect_true(all(preds$.pred_lower <= preds$.pred))
  expect_true(all(preds$.pred <= preds$.pred_upper))

  score <- yardstick::rmse_vec(truth = future$y, estimate = preds$.pred)
  expect_true(is.finite(score))
})

test_that("predictions are row-aligned to new_data across multiple series", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("distributional")

  set.seed(2)
  train <- rbind(
    data.frame(store = "a", t = 1:20, y = cumsum(rnorm(20)) + 100),
    data.frame(store = "b", t = 1:20, y = cumsum(rnorm(20)) + 50)
  )
  # Interleaved series order, to prove the scatter-back is correct.
  future <- data.frame(
    store = c("b", "a", "b", "a"),
    t     = c(21L, 21L, 22L, 22L)
  )

  model <- tsfm_pretrained("stub")
  fit <- tsfm_fit(y ~ 1, data = train, model = model, index = "t", id = "store")
  preds <- predict(fit, new_data = future)

  expect_identical(nrow(preds), nrow(future))
  # Series "a" starts near 100+, "b" near 50 — the point forecast is the last
  # observed value, so row order must follow `future`, not sorted order.
  a_last <- tail(train$y[train$store == "a"], 1)
  b_last <- tail(train$y[train$store == "b"], 1)
  expect_equal(preds$.pred[future$store == "a"], rep(a_last, 2))
  expect_equal(preds$.pred[future$store == "b"], rep(b_last, 2))
})

test_that("unknown series in new_data are rejected", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("distributional")

  train <- data.frame(store = "a", t = 1:10, y = as.numeric(1:10))
  model <- tsfm_pretrained("stub")
  fit <- tsfm_fit(y ~ 1, data = train, model = model, index = "t", id = "store")
  future <- data.frame(store = "z", t = 11:12)
  expect_error(predict(fit, new_data = future), "No fitted history")
})
