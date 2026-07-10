test_that("the stub architecture is registered at load", {
  expect_true(tsfm_registry_has("stub"))
  expect_true("stub" %in% tsfm_registry_archs())
  expect_type(tsfm_registry_get("stub"), "closure")
})

test_that("registration guards against silent clobbering", {
  ctor <- function(config, weights) NULL
  tsfm_register_arch("test-arch", ctor)
  on.exit(rm("test-arch", envir = tsfm:::.tsfm_registry), add = TRUE)
  expect_error(tsfm_register_arch("test-arch", ctor), "already registered")
  expect_silent(tsfm_register_arch("test-arch", ctor, overwrite = TRUE))
})

test_that("unknown architectures error helpfully", {
  expect_error(tsfm_registry_get("no-such-arch"), "No constructor registered")
})
