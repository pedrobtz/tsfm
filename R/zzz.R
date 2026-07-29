.onLoad <- function(libname, pkgname) {
  # Register the built-in architectures. Third parties call tsfm_register_arch()
  # from their own .onLoad to extend the catalogue without touching this package.
  tsfm_register_arch("stub", stub_constructor, overwrite = TRUE)
  tsfm_register_arch("ttm", ttm_constructor, overwrite = TRUE)
  tsfm_register_arch("timesfm", timesfm_constructor, overwrite = TRUE)
  tsfm_register_arch("chronos2", chronos2_constructor, overwrite = TRUE)

  # Register the parsnip engine when parsnip is available, mirroring how
  # brulee/modeltime engines register lazily so the core package stays light.
  # Best-effort: registration must never block package load.
  if (requireNamespace("parsnip", quietly = TRUE)) {
    tryCatch(make_tsfm_reg(), error = function(e) NULL)
  }
}
