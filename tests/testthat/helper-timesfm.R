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
