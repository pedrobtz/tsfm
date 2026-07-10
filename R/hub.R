#' Load a time-series foundation model from the Hugging Face Hub
#'
#' Resolves a model's configuration, looks up the matching architecture
#' constructor in the registry, and returns a uniform [new_tsfm_model()] handle.
#'
#' In Stage 0 only the built-in `"stub"` model is fully wired: passing
#' `"stub"` (or any id beginning with `"stub"`) synthesises a config in memory
#' and skips the download entirely, so the loader/registry/forecast contract can
#' be exercised without network access or torch. For real checkpoints the
#' function downloads `config.json` through \pkg{hfhub} (with the revision
#' pinned), reads the architecture key, and dispatches to the registry; the
#' native architectures that consume the downloaded weights arrive from Stage 1.
#'
#' @param model_id Character scalar. A Hub repo id such as
#'   `"ibm-granite/granite-timeseries-ttm-r2"`, or `"stub"` for the built-in
#'   Stage 0 model.
#' @param revision Character scalar. A branch, tag, or commit SHA to pin.
#'   Defaults to `"main"`; pinning to a SHA is recommended for reproducibility.
#' @param ... Passed to the architecture constructor.
#' @return A [new_tsfm_model()].
#' @export
tsfm_pretrained <- function(model_id, revision = "main", ...) {
  model_id <- as.character(model_id)
  resolved <- tsfm_resolve_config(model_id, revision, ...)
  constructor <- tsfm_registry_get(resolved$config$architecture)
  constructor(resolved$config, resolved$weights)
}

# Distinguish the built-in stub from a real Hub id.
is_stub_id <- function(model_id) {
  identical(model_id, "stub") || grepl("^stub([-/]|$)", model_id)
}

# Return list(config = <parsed config, incl. `architecture`>, weights = <state
# dict or NULL>). The stub branch is self-contained; the real branch is the
# hfhub plumbing that Stage 1's native architectures build on.
tsfm_resolve_config <- function(model_id, revision, ..., call = rlang::caller_env()) {
  if (is_stub_id(model_id)) {
    config <- list(
      architecture = "stub",
      model_id     = model_id,
      revision     = revision,
      max_context  = 512L
    )
    return(list(config = config, weights = NULL))
  }

  # Real checkpoints: download and parse config.json, normalise the architecture
  # key, then (Stage 1+) hand off to a registered native constructor. Until a
  # native architecture is registered, `tsfm_pretrained()` will stop at the
  # registry lookup with a clear "not registered" error rather than downloading
  # weights it cannot yet consume, so we defer the weight fetch to the
  # constructor path in a later stage.
  rlang::check_installed("hfhub", reason = "to download checkpoints from the Hugging Face Hub.")
  config_path <- hfhub::hub_download(model_id, "config.json", revision = revision)
  config <- jsonlite_read(config_path)
  config$architecture <- normalize_architecture(config)
  config$model_id <- model_id
  config$revision <- revision
  list(config = config, weights = NULL)
}

# Read a JSON config without taking a hard dependency: hfhub pulls in jsonlite,
# but resolve it at call time to keep this decoupled.
jsonlite_read <- function(path) {
  rlang::check_installed("jsonlite", reason = "to parse checkpoint configs.")
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

# Map a Hub config to our architecture key. Hugging Face configs expose either a
# top-level `architectures` array (transformers convention) or a `model_type`
# field; different TSFMs use different conventions, so both are handled.
normalize_architecture <- function(config) {
  arch <- config$architecture %||% config$model_type %||% config$architectures[1]
  if (is.null(arch) || is.na(arch)) {
    cli::cli_abort("Could not determine the architecture from the checkpoint config.")
  }
  tolower(as.character(arch)[1])
}
