import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


# Splits the vocab table across GPUs by token ID range.
# GPU 0 owns token IDs [0, vocab/tp), GPU 1 owns [vocab/tp, 2*vocab/tp), etc.
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int, # num_embeddings = vocab size 
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank  # first token ID this GPU owns
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader # attach a custom weight loading function to this parameter, pointing to the class's weight_loader method. 

    # Load this GPU's shard of the vocab table from the checkpoint
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data # size = (vocab/tp, embedding_dim)
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size) # loaded_weight[start_idx : start_idx + shard_size, :]
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)  # which tokens belong to this GPU
            x = mask * (x - self.vocab_start_idx)  # convert global IDs to local indices; out-of-range → 0 (safe dummy)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y  # zero out embeddings for tokens this GPU doesn't own
            dist.all_reduce(y)         # sum across GPUs → each token gets its real embedding (only one GPU was non-zero)
        return y


# Reverse of VocabParallelEmbedding: hidden states → logits over vocab.
# Reuses the same weight (tied embeddings — input and output share vocab table).
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # x has shape (total_uncached_tokens, hidden_size) during prefill.
            # Only the last token of each sequence needs to produce a logit.
            last_indices = context.cu_seqlens_q[1:] - 1  # index of last token per sequence in flat packed tensor
            x = x[last_indices].contiguous()
        # decode: x (bs, hidden_size), produce logits for the last token of each sequence
        logits = F.linear(x, self.weight)  # (bs, vocab_size_per_gpu) — each GPU has partial vocab logits
        if self.tp_size > 1:
            # gather (not all-reduce): only rank 0 needs full logits for sampling
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None  # (bs, full_vocab_size) on rank 0 only
        return logits
