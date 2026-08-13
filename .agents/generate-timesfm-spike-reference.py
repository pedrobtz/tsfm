"""Generate the Stage 1 reference from pinned official TimesFM source.

Run with the source checkout at TIMESFM_SOURCE and commit
3dae50b20d7a724981e8ea36cda75578f80dd2dc. The script prints the embedding and
continuous-quantile-head values embedded in test-timesfm-feasibility.R.
"""

import os
import sys

import torch

source = os.environ["TIMESFM_SOURCE"]
sys.path.insert(0, os.path.join(source, "src"))

from timesfm import configs  # noqa: E402
from timesfm.torch import dense, transformer  # noqa: E402


def values(shape, offset, scale=0.05, base=0.0):
  n = 1
  for size in shape:
    n *= size
  return (base + scale * torch.sin(torch.arange(n) + offset)).reshape(shape)


cfg = configs.TransformerConfig(
    model_dims=4,
    hidden_dims=4,
    num_heads=2,
    attention_norm="rms",
    feedforward_norm="rms",
    qk_norm="rms",
    use_bias=False,
    use_rotary_position_embeddings=True,
    ff_activation="swish",
    fuse_qkv=True,
)
block = transformer.Transformer(cfg)
head = dense.ResidualBlock(
    configs.ResidualBlockConfig(
        input_dims=4,
        hidden_dims=4,
        output_dims=9,
        use_bias=False,
        activation="swish",
    )
)

with torch.no_grad():
  block.pre_attn_ln.scale.copy_(values((4,), 1, 0.02, 1))
  block.post_attn_ln.scale.copy_(values((4,), 2, 0.02, 1))
  block.attn.qkv_proj.weight.copy_(values((12, 4), 3))
  block.attn.query_ln.scale.copy_(values((2,), 4, 0.02, 1))
  block.attn.key_ln.scale.copy_(values((2,), 5, 0.02, 1))
  block.attn.per_dim_scale.per_dim_scale.copy_(values((2,), 6, 0.1))
  block.attn.out.weight.copy_(values((4, 4), 7))
  block.pre_ff_ln.scale.copy_(values((4,), 8, 0.02, 1))
  block.ff0.weight.copy_(values((4, 4), 9))
  block.ff1.weight.copy_(values((4, 4), 10))
  block.post_ff_ln.scale.copy_(values((4,), 11, 0.02, 1))
  head.hidden_layer.weight.copy_(values((4, 4), 12))
  head.output_layer.weight.copy_(values((9, 4), 13))
  head.residual_layer.weight.copy_(values((9, 4), 14))

x = torch.tensor(
    [[[0.2, -0.1, 0.4, 0.3], [0.5, 0.7, -0.2, 0.1], [-0.4, 0.6, 0.8, -0.3]]],
    dtype=torch.float32,
)
mask = torch.tensor([[False, False, False]])
with torch.no_grad():
  embedding, _ = block(x, mask)
  quantiles = head(embedding).reshape(1, 3, 3, 3)

torch.set_printoptions(precision=10, linewidth=200, sci_mode=False)
print("EMBEDDINGS")
print(embedding.flatten())
print("QUANTILES")
print(quantiles.flatten())
