# `torch` ships as two pieces: the R package, and the LibTorch runtime that
# `torch::install_torch()` downloads separately. `skip_if_not_installed("torch")`
# only proves the first, so any test that actually executes a tensor needs this
# instead --- otherwise it fails with "Lantern is not loaded" on a machine, or a
# CI runner, where the runtime was never fetched.
skip_if_no_torch <- function() {
  testthat::skip_if_not_installed("torch")
  testthat::skip_if_not(torch::torch_is_installed(), "LibTorch is not installed")
}
