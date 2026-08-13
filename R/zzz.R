utils::globalVariables(c("object", "new_data"))

.onLoad <- function(libname, pkgname) {
  # Register the built-in architectures. Third parties call tsfm_register_arch()
  # from their own .onLoad to extend the catalogue without touching this package.
  tsfm_register_arch("stub", stub_constructor, overwrite = TRUE)
  tsfm_register_arch("ttm", ttm_constructor, overwrite = TRUE)
  tsfm_register_arch("timesfm", timesfm_constructor, overwrite = TRUE)

  # Chronos-2 is intentionally not registered for 0.1.0. The existing Brulee
  # bridge remains in the source tree as reference work, but it is not a
  # verified implementation of the architecture contract and must not appear
  # as an executable built-in model.

  # Adapter methods are registered against generics owned by optional packages.
  # s3_register() installs a load hook, so the method lights up if and when the
  # adapter package is loaded, and costs nothing when it never is. This is what
  # lets tsfm interoperate with fabletools without importing it — and without
  # defining a competing `as_fable()` generic.
  vctrs::s3_register("fabletools::as_fable", "tsfm_forecast")

  # Register the parsnip engine when parsnip is available, mirroring how
  # brulee/modeltime engines register lazily so the core package stays light.
  # Best-effort: registration must never block package load.
  if (requireNamespace("parsnip", quietly = TRUE)) {
    tryCatch(
      make_tsfm_reg(),
      error = function(e) {
        cli::cli_warn(c(
          "Could not register the {.val tsfm} parsnip engine.",
          "i" = "The engine will be unavailable; everything else still works.",
          "x" = conditionMessage(e)
        ))
      }
    )
  }
}
