test_that("the safe catalogue default only returns supported checkpoints", {
  local_mocked_bindings(
    tsfm_probe_cached_file = function(...) FALSE
  )
  supported <- tsfm_models()
  all <- tsfm_models(state = NULL)

  expect_identical(nrow(supported), 0L)
  expect_identical(nrow(all), 2L)
  expect_setequal(unique(all$state), "scaffold")
  expect_true(all(grepl("^[0-9a-f]{40}$", all$revision)))
  expect_true(is.list(all$quantile_levels))
  expect_false(any(all$multivariate))
  expect_false(any(all$past_covariates))
  expect_false(any(all$future_covariates))
  expect_setequal(
    names(all),
    c(
      "model_id", "architecture", "revision", "state", "max_context",
      "quantile_levels", "multivariate", "past_covariates",
      "future_covariates", "n_params", "size_bytes", "cached", "license"
    )
  )
})

test_that("manifest cache state distinguishes complete, incomplete, and unknown", {
  record <- tsfm_catalogue_records()[[1]]
  expect_true(tsfm_manifest_cached(record, function(...) TRUE))

  probe <- function(model_id, revision, file) !identical(file, "model.safetensors")
  expect_false(tsfm_manifest_cached(record, probe))

  record$manifest <- NULL
  expect_true(is.na(tsfm_manifest_cached(record, function(...) stop("not called"))))
})

test_that("catalogue cache probes are local-only and reflected per manifest", {
  calls <- list()
  local_mocked_bindings(
    tsfm_probe_cached_file = function(model_id, revision, file) {
      calls[[length(calls) + 1L]] <<- list(model_id, revision, file)
      !identical(file, "model.safetensors")
    }
  )
  out <- tsfm_models(NULL)
  expect_false(any(out$cached))
  expect_length(calls, 4L)
})

test_that("hfhub cache probes explicitly disable outgoing traffic", {
  seen <- NULL
  local_mocked_bindings(
    hub_download = function(...) {
      seen <<- list(...)
      tempfile()
    },
    .package = "hfhub"
  )
  expect_true(tsfm_probe_cached_file("org/model", "abc", "config.json"))
  expect_true(seen$local_files_only)
  expect_identical(seen$repo_id, "org/model")
  expect_identical(seen$revision, "abc")
})

test_that("invalid catalogue states are structured recoverable errors", {
  error <- expect_error(tsfm_models("planned"), class = "tsfm_error_capability")
  expect_s3_class(error, "tsfm_error_recoverable")
  expect_identical(error$capability, "catalogue_state")
})

test_that("scaffold checkpoints fail before download", {
  local_mocked_bindings(
    tsfm_resolve_config = function(...) stop("download path was reached")
  )
  error <- expect_error(
    tsfm_pretrained("google/timesfm-2.5-200m-pytorch"),
    class = "tsfm_error_capability"
  )
  expect_identical(error$capability, "model_state")
  expect_identical(error$requested, "scaffold")
})
