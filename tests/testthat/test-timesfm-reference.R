timesfm_reference_files <- function() {
  list.files(
    testthat::test_path("fixtures", "timesfm"),
    pattern = "\\.json$",
    full.names = TRUE
  )
}

test_that("TimesFM golden fixtures cover the Stage 2 reference cases", {
  fixtures <- lapply(
    timesfm_reference_files(),
    jsonlite::fromJSON,
    simplifyVector = TRUE
  )
  names(fixtures) <- vapply(fixtures, `[[`, character(1), "name")

  expect_setequal(
    names(fixtures),
    c("typical", "short_context", "context_truncation", "batch_agreement")
  )
  for (fixture in fixtures) {
    expect_identical(
      fixture$source_commit,
      "3dae50b20d7a724981e8ea36cda75578f80dd2dc"
    )
    expect_identical(
      fixture$revision,
      "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    )
    expect_equal(fixture$quantile_levels, seq(0.1, 0.9, by = 0.1))
    expect_length(fixture$context_files, nrow(fixture$context_specs))
    expect_true(all(file.exists(testthat::test_path(
      "fixtures", "timesfm", unlist(fixture$context_files)
    ))))
    n_contexts <- nrow(fixture$context_specs)
    expect_identical(dim(fixture$expected_point)[[1]], n_contexts)
    expect_identical(dim(fixture$expected_point)[[2]], fixture$horizon)
    expect_identical(dim(fixture$expected_quantiles), c(
      n_contexts, fixture$horizon, 9L
    ))
    # Each fixture records the comparison budget it was generated against, as
    # an atol/rtol pair rather than one absolute number.
    expect_equal(fixture$atol, 1e-4)
    expect_equal(fixture$rtol, 1e-5)
    median <- fixture$expected_quantiles[, , 5]
    if (n_contexts == 1L) median <- matrix(median, nrow = 1L)
    expect_close_f32(
      median,
      fixture$expected_point,
      atol = fixture$atol, rtol = fixture$rtol
    )
  }
  expect_identical(fixtures$short_context$context_specs$values[[1]], c(3, 4.5, 4))
  expect_identical(fixtures$context_truncation$max_context, 16256L)
  expect_gt(fixtures$context_truncation$context_specs$length[[1]], 16256L)
  expect_lte(fixtures$batch_agreement$batch_loop_max_abs_difference, 1e-4)
})

test_that("the reference generator and lock file carry every immutable pin", {
  generator_path <- testthat::test_path(
    "..", "..", ".agents", "generate-timesfm-reference.py"
  )
  requirements_path <- testthat::test_path(
    "..", "..", ".agents", "timesfm-reference-requirements.txt"
  )
  skip_if_not(file.exists(generator_path) && file.exists(requirements_path))
  generator <- readLines(generator_path)
  requirements <- readLines(requirements_path)
  expect_true(any(grepl("3dae50b20d7a724981e8ea36cda75578f80dd2dc", generator)))
  expect_true(any(grepl("1d952420fba87f3c6dee4f240de0f1a0fbc790e3", generator)))
  expect_true(any(grepl("torch==2.2.2", requirements, fixed = TRUE)))
  expect_true(any(grepl("safetensors==0.5.3", requirements, fixed = TRUE)))
  expect_true(any(grepl("huggingface_hub==0.36.0", requirements, fixed = TRUE)))
})
