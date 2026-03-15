import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # scale by temperature
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) # float() converts to full-precision. div_() is in-place divisition
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1) # Gumbel-max sampling
        return sample_tokens
