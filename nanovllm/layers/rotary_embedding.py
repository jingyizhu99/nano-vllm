from functools import lru_cache
import torch
from torch import nn


# Apply rotary position embedding to a single tensor (Q or V).
# Splits the head vector in half and applies a 2D rotation to each pair:
#   [y1]   [cos  -sin] [x1]
#   [y2] = [sin   cos] [x2]
# The rotation angle is position-dependent (baked into cos/sin),
# so tokens at different positions get rotated differently.
# This makes Q·K attention scores naturally encode relative position.
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)  # split head_dim in half
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,         # base frequency, typically 10000 or 500000
    ) -> None:
        super().__init__()
        self.head_size = head_size # = head_dim
        assert rotary_dim == head_size
        # rotary_dim/2 geometrically spaced frequencies
        # low index = high freq (sensitive to nearby positions)
        # high index = low freq (sensitive to distant positions)
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)  # [0, 1, 2, ..., max_pos-1]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # outer product: (max_pos, rotary_dim/2)
        cos = freqs.cos()
        sin = freqs.sin()
        # precompute and cache cos/sin for all positions upfront
        # shape: (max_positions, 1, rotary_dim) — the 1 allows broadcasting over heads
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # register as non-trainable buffer: moves to GPU with model, not saved in checkpoints
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,   # absolute position of each token — critical for decode where pos != 0
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]  # lookup precomputed cos/sin for each token's position
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)    # V is never rotated — only Q and K affect attention scores
        return query, key


# Cached factory — lru_cache(1) returns the same instance if called with same args.
# All attention layers share the same RoPE config so this avoids recomputing
# the cos/sin table for every layer.
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
