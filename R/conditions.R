# Structured engine conditions.
#
# Consumers make policy decisions from these classes and fields. Every engine
# abort has one leaf class, exactly one policy parent, and the shared
# `tsfm_error` base; messages remain for humans rather than control flow.

tsfm_error_policies <- c(
  "tsfm_error_recoverable",
  "tsfm_error_external",
  "tsfm_error_internal"
)

tsfm_require_namespace <- function(package, reason = NULL,
                                   call = rlang::caller_env()) {
  missing <- package[!vapply(package, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    detail <- if (is.null(reason)) "" else paste0(" ", reason)
    tsfm_abort_capability(
      c(
        "Required package{?s} {.pkg {missing}} {?is/are} not installed.",
        "i" = "Install {?it/them} with {.code install.packages({deparse(missing)})}.{detail}"
      ),
      capability = "package_dependency",
      requested = missing,
      supported = package,
      call = call
    )
  }
  invisible(package)
}

tsfm_abort <- function(message, class, policy, ...,
                       call = rlang::caller_env(),
                       .envir = rlang::caller_env()) {
  stopifnot(
    length(class) == 1L,
    length(policy) == 1L,
    policy %in% tsfm_error_policies
  )
  cli::cli_abort(
    message,
    class = c(class, policy, "tsfm_error"),
    ...,
    call = call,
    .envir = .envir
  )
}

tsfm_abort_capability <- function(message,
                                  model_id = NA_character_,
                                  revision = NA_character_,
                                  capability,
                                  requested = NULL,
                                  supported = NULL,
                                  call = rlang::caller_env(),
                                  .envir = rlang::caller_env()) {
  tsfm_abort(
    message,
    class = "tsfm_error_capability",
    policy = "tsfm_error_recoverable",
    model_id = as.character(model_id),
    revision = as.character(revision),
    capability = as.character(capability),
    requested = requested,
    supported = supported,
    call = call,
    .envir = .envir
  )
}

tsfm_abort_context_length <- function(message,
                                      model_id = NA_character_,
                                      revision = NA_character_,
                                      requested,
                                      supported,
                                      call = rlang::caller_env(),
                                      .envir = rlang::caller_env()) {
  tsfm_abort(
    message,
    class = "tsfm_error_context_length",
    policy = "tsfm_error_recoverable",
    model_id = as.character(model_id),
    revision = as.character(revision),
    requested = requested,
    supported = supported,
    call = call,
    .envir = .envir
  )
}

tsfm_abort_quantile_levels <- function(message,
                                       model_id = NA_character_,
                                       revision = NA_character_,
                                       requested,
                                       supported,
                                       call = rlang::caller_env(),
                                       .envir = rlang::caller_env()) {
  tsfm_abort(
    message,
    class = "tsfm_error_quantile_levels",
    policy = "tsfm_error_recoverable",
    model_id = as.character(model_id),
    revision = as.character(revision),
    requested = requested,
    supported = supported,
    call = call,
    .envir = .envir
  )
}

tsfm_abort_device <- function(message,
                              requested_device,
                              resolved_device = NA_character_,
                              call = rlang::caller_env(),
                              .envir = rlang::caller_env()) {
  tsfm_abort(
    message,
    class = "tsfm_error_device",
    policy = "tsfm_error_recoverable",
    requested_device = as.character(requested_device),
    resolved_device = as.character(resolved_device),
    call = call,
    .envir = .envir
  )
}

tsfm_abort_download <- function(message,
                                model_id,
                                revision,
                                file = NA_character_,
                                parent = NULL,
                                call = rlang::caller_env(),
                                .envir = rlang::caller_env()) {
  tsfm_abort(
    message,
    class = "tsfm_error_download",
    policy = "tsfm_error_external",
    model_id = as.character(model_id),
    revision = as.character(revision),
    file = as.character(file),
    parent = parent,
    call = call,
    .envir = .envir
  )
}

tsfm_abort_checkpoint <- function(message,
                                  model_id = NA_character_,
                                  revision = NA_character_,
                                  tensor = NA_character_,
                                  expected = NULL,
                                  actual = NULL,
                                  call = rlang::caller_env(),
                                  .envir = rlang::caller_env()) {
  tsfm_abort(
    message,
    class = "tsfm_error_checkpoint",
    policy = "tsfm_error_internal",
    model_id = as.character(model_id),
    revision = as.character(revision),
    tensor = as.character(tensor),
    expected = expected,
    actual = actual,
    call = call,
    .envir = .envir
  )
}

tsfm_abort_contract <- function(message,
                                architecture = NA_character_,
                                model_id = NA_character_,
                                contract = NA_character_,
                                expected = NULL,
                                actual = NULL,
                                call = rlang::caller_env(),
                                .envir = rlang::caller_env()) {
  tsfm_abort(
    message,
    class = "tsfm_error_contract",
    policy = "tsfm_error_internal",
    architecture = as.character(architecture),
    model_id = as.character(model_id),
    contract = as.character(contract),
    expected = expected,
    actual = actual,
    call = call,
    .envir = .envir
  )
}
