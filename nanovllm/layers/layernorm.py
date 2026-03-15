import torch
from torch import nn


# RMSNorm: normalizes hidden states by their root mean square.
# Formula: x / sqrt(mean(x²) + eps) * weight
# Simpler than LayerNorm (no mean subtraction) — used by Llama, Qwen, etc.
# Keeps activations in a consistent range across transformer layers for stability.
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,   # small constant to prevent division by zero
    ) -> None:
        super().__init__()
        self.eps = eps

        # This is the learned scale of RMSNorm. (γ)
        # initialized to 1 (identity)
        # shape = (hidden_size,)
        # Pure normalization would force all hidden states to have RMS=1, which might destroy information the model learned.
        # So the weight parameter lets the model undo normalization for specific dimensions if needed
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # Plain RMSNorm — used for the first norm in the model (no residual yet)
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()  # upcast to float32 for numerical stability
        var = x.pow(2).mean(dim=-1, keepdim=True)  # mean(x²) per token, shape (N, 1)
        x.mul_(torch.rsqrt(var + self.eps))         # divide by sqrt(mean(x²) + eps)
        x = x.to(orig_dtype).mul_(self.weight)      # scale by learned weight, cast back to bfloat16
        return x

    # Fused residual add + RMSNorm — used for all subsequent norms in the model.
    # x = norm(x + residual), where residual is a running sum, residual is from the previous layer = prev_x + pre_residual
    # Fuses both ops to save a memory read/write.
    # Returns both the normalized x (for next sublayer) and the updated residual (for next norm).
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())  # x = sublayer_output + residual (running sum)
        residual = x.to(orig_dtype)           # save pre-norm sum as new residual for the next layer
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual  # x = normalized (fed into next sublayer), residual = unnormalized (carried forward)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)         # first norm: no residual yet
        else:
            return self.add_rms_forward(x, residual)  # all other norms: fuse residual add
