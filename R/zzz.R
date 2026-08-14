utils::globalVariables(c("object", "new_data"))

.onLoad <- function(libname, pkgname) {
  # Register the built-in architectures. Third parties call tsfm_register_arch()
  # from their own .onLoad to extend the catalogue without touching this package.
  tsfm_register_arch("stub", stub_constructor, overwrite = TRUE)
  tsfm_register_arch("ttm", ttm_constructor, overwrite = TRUE)
  tsfm_register_arch("timesfm", timesfm_constructor, overwrite = TRUE)

  # Chronos-2 is intentionally not registered for 0.1.0. The Brulee bridge is
  # kept as prior art in .agents/reference/arch-chronos2.R rather than shipped:
  # it is not a verified implementation of the architecture contract, so it must
  # not appear as an executable built-in model, and the package must not carry a
  # dependency for code nothing calls. tsfm_pretrained() rejects Chronos-2 ids
  # before any network or tensor work; see is_chronos2_id().

  # Adapter methods are registered against generics owned by optional packages.
  # s3_register() installs a load hook, so the method lights up if and when the
  # adapter package is loaded, and costs nothing when it never is. This is what
  # lets tsfm interoperate with fabletools without importing it — and without
  # defining a competing `as_fable()` generic.
  vctrs::s3_register("fabletools::as_fable", "tsfm_forecast")
  # `model_sum()` is fabletools' own generic, so it can only be registered
  # lazily. `forecast()` needs no hook: it comes from generics, which tsfm
  # imports and re-exports, and fabletools dispatches on that same generic.
  vctrs::s3_register("fabletools::model_sum", "model_tsfm")

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

  # Register the harus backend when harus is loaded. The registry is optional:
  # if harus is not installed, nothing happens. tsfm works standalone. If both
  # are loaded, this makes every pretrained checkpoint available in harus'
  # model registry without requiring harus to know tsfm's internals.
  if (requireNamespace("harus", quietly = TRUE)) {
    tryCatch(
      tsfm_register_harus_backend(),
      error = function(e) {
        # Silent: harus is optional, and its absence is normal. The message is
        # for debugging only.
        if (identical(Sys.getenv("TSFM_VERBOSE_LOAD"), "true")) {
          message("tsfm: could not register harus backend: ", conditionMessage(e))
        }
      }
    )
  }
}
