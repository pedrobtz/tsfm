# Stage 1 CPU feasibility benchmark for the exact TimesFM 2.5 block dimensions.
# Run from the repository root with a working R torch/LibTorch installation.

devtools::load_all(".", quiet = TRUE)
torch::torch_manual_seed(20260813)

model_dim <- 1280L
num_heads <- 16L
n_patches <- 512L # 16,384 context values / patch length 32

module <- timesfm_spike_module(
  model_dim = model_dim,
  num_heads = num_heads,
  hidden_dim = model_dim,
  quantile_horizon = 1024L,
  n_outputs = 10L
)
module$eval()

parameters <- sum(vapply(
  module$parameters,
  function(x) as.numeric(x$numel()),
  numeric(1)
))
input <- torch::torch_randn(c(1, n_patches, model_dim))
mask <- torch::torch_zeros(c(1, n_patches), dtype = torch::torch_bool())

elapsed <- system.time({
  result <- torch::with_no_grad({
    embedding <- module$transformer(input, mask)
    # The release path only consumes the last patch's continuous head. This
    # avoids materializing unused 1024x10 predictions for every context patch.
    last <- embedding$narrow(2, n_patches, 1)
    quantiles <- module$quantile_head(last)$reshape(c(1, 1, 1024, 10))
    list(embedding = embedding, quantiles = quantiles)
  })
})

cat("parameters:", parameters, "\n")
cat("parameter_bytes_f32:", parameters * 4, "\n")
cat("embedding_shape:", paste(result$embedding$shape, collapse = "x"), "\n")
cat("quantile_shape:", paste(result$quantiles$shape, collapse = "x"), "\n")
cat("elapsed_seconds:", unname(elapsed[["elapsed"]]), "\n")
