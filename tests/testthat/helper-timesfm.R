# A tiny TimesFM of the same shape as the real one: same layer types, same
# weight map, same decode path, ~31k parameters instead of 231M. Structural
# invariants --- flip antisymmetry, batch/loop agreement, determinism, the
# positivity clamp --- hold for any weights, so they can be checked here on
# every platform instead of only where the 925 MB checkpoint is cached.
timesfm_synthetic_config <- function() {
  list(
    architecture = "timesfm",
    model_id = "synthetic/timesfm",
    revision = strrep("b", 40L),
    device = "cpu",
    hidden_size = 32L,
    intermediate_size = 32L,
    head_dim = 8L,
    num_attention_heads = 4L,
    num_hidden_layers = 2L,
    patch_length = 4L,
    horizon_length = 8L,
    quantile_horizon_length = 32L,
    context_length = 64L,
    quantiles = seq(0.1, 0.9, by = 0.1),
    rms_norm_eps = 1e-6
  )
}

timesfm_synthetic_module <- function(config = timesfm_synthetic_config(),
                                     seed = 20260814L) {
  torch::torch_manual_seed(seed)
  spec <- timesfm_expected_state_spec(config)
  # Small weights keep the untrained forward pass in a numerically sane range.
  weights <- lapply(spec, function(shape) torch::torch_randn(shape) * 0.05)
  module <- timesfm_module(config)
  timesfm_load_module_weights(module, weights, config)
  module$eval()
  module
}

timesfm_test_config <- function() {
  list(
    architecture = "timesfm",
    model_id = "fixture/timesfm",
    revision = strrep("a", 40L),
    device = "cpu",
    hidden_size = 1280L,
    intermediate_size = 1280L,
    head_dim = 80L,
    num_attention_heads = 16L,
    num_hidden_layers = 20L,
    patch_length = 32L,
    horizon_length = 128L,
    quantile_horizon_length = 1024L,
    context_length = 16384L,
    quantiles = seq(0.1, 0.9, by = 0.1),
    rms_norm_eps = 1e-6
  )
}
