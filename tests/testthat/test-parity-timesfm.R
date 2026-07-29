# Numerical-parity harness for the native TimesFM port (roadmap Stage 2 exit
# gate). Identical in spirit to test-parity-ttm.R: fixed inputs vs. the
# reference implementation's outputs, committed as data so CI needs neither
# Python nor the Hub. Skips until the numerical port lands and fixtures exist.

test_that("native TimesFM matches the reference on golden fixtures", {
  skip_if_not_installed("torch")
  skip_if_not_installed("jsonlite")
  dir <- testthat::test_path("fixtures", "timesfm")
  skip_if_not(dir.exists(dir) && length(list.files(dir, pattern = "\\.json$")) > 0,
              "No TimesFM golden fixtures yet; generate them per fixtures/README.md.")

  model <- tsfm_pretrained("google/timesfm-2.5-200m-pytorch")

  files <- list.files(dir, pattern = "\\.json$", full.names = TRUE)
  for (f in files) {
    fx <- jsonlite::fromJSON(f, simplifyVector = TRUE)
    context <- as.numeric(fx$context)
    levels <- as.numeric(fx$quantile_levels)
    h <- length(fx$expected_median)

    q <- model$predict_fn(context, h, levels)
    median_col <- match(0.5, levels)

    expect_equal(q[, median_col], as.numeric(fx$expected_median),
                 tolerance = fx$tolerance %||% 1e-4)
    if (!is.null(fx$expected_quantiles)) {
      expected <- matrix(as.numeric(unlist(fx$expected_quantiles)),
                         nrow = h, ncol = length(levels), byrow = TRUE)
      expect_equal(unname(q), expected, tolerance = fx$tolerance %||% 1e-4)
    }
  }
})
