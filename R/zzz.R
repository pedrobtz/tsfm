.onLoad <- function(libname, pkgname) {
  # Register the built-in architectures. Third parties call tsfm_register_arch()
  # from their own .onLoad to extend the catalogue without touching this package.
  tsfm_register_arch("stub", stub_constructor, overwrite = TRUE)
}
