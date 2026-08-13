# Numerical closeness for float32 ports.
#
# A bare `max(abs(a - b)) < tol` is the wrong shape for this comparison: one
# absolute number is simultaneously too tight for large values and too loose for
# small ones. The ecosystem convention --- `torch.testing.assert_close()`, which
# upstream TimesFM's own tests use --- is a combined criterion:
#
#     |actual - expected| <= atol + rtol * |expected|
#
# The defaults below are PyTorch's documented float32 pair. Tolerance here is set
# by float32 accumulation, not by correctness: reassociation across LibTorch
# builds and BLAS backends moves results by single-digit to low-tens of ulps
# (one ulp is about 1.2e-7 relative), while a structural error --- a transposed
# weight, a dropped rotation, a wrong channel --- moves them by 1e-1 or more.
expect_close_f32 <- function(actual, expected, atol = 1e-5, rtol = 1.3e-6) {
  actual <- as.numeric(actual)
  expected <- as.numeric(expected)
  testthat::expect_identical(length(actual), length(expected))
  budget <- atol + rtol * abs(expected)
  # Reporting the worst diff as a fraction of its budget makes the failure
  # message say how much headroom was left, not just that a threshold was hit.
  testthat::expect_lte(max(abs(actual - expected) / budget), 1)
}
