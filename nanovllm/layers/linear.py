import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim # which dimension of the weight matrix to split across GPUs.
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader # attach a custom weight loading function to this parameter
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# no tensor parallelism
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

   # Each subclass defines its own weight_loader to handle how its shard of the weight matrix gets loaded.
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight) # copies the values from loaded_weight into the memory already allocated for param.data，CPU → GPU transfer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


'''
  Weight matrix split along columns (output dimension) across GPUs:

  Full weight: (output_size, input_size)
  GPU 0 holds: (output_size/tp_size, input_size)   ← first chunk of rows
  GPU 1 holds: (output_size/tp_size, input_size)   ← second chunk of rows
  No all-reduce needed — each GPU produces a partial output that gets used independently downstream.
'''
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data # size = (output_size/tp_size, input_size)
        shard_size = param_data.size(self.tp_dim) # self.tp_dim = 0; shard_size = output_size // tp_size
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size) # equivalent to: loaded_weight[start_idx : start_idx + shard_size, :]
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


'''
Used when multiple weight matrices are fused into one (e.g. gate + up projections in MLP).
Each sub-projection gets its own slice of the parameter
原本的做法： GPU 要先算一次 Gate，再算一次 Up。这需要启动两个 CUDA Kernel，搬运两次数据
合并的做法： 既然 Gate 和 Up 的输入都是一样的 ($x$)，且形状也相同，我们干脆把它们的权重矩阵“上下拼接”成一个大矩阵
性能提升： GPU 只需要做一次矩阵乘法（GEMM），就能同时拿到 Gate 和 Up 的结果。然后代码里再用 chunk(2) 把结果切开分别处理
'''
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data # size = (sum(output_sizes)/tp_size, input_size)
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size # 确定这个 Shard 在合并后矩阵（param_data）中的起始位置 (纵向偏移)
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size # 确定这个 Shard 在 GPU 上应该占多大空间
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size) # 从合并后的矩阵中切出这个 Shard 应该占的那一块（纵向切片）
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank] # 从磁盘读取的原始权重中，切出属于当前 GPU Rank 的那一部分
        param_data.copy_(loaded_weight)


'''
Specialized for attention — Q, K, V are stored as one weight matrix in the checkpoint but have different sizes
(Q has more heads than K,V in GQA)
'''
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size) # number of Q heads per GPU
        self.num_kv_heads = divide(total_num_kv_heads, tp_size) # number of K,V heads per GPU
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        # 在param_data中的拼接顺序是q, k, v
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size) # equivalent to: param_data[shard_offset : shard_offset + shard_size, :]
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank] # self.tp_size：切成几块；self.tp_dim：切哪个维度；self.tp_rank：取第几块
        param_data.copy_(loaded_weight)


'''
Weight matrix split along rows (input dimension):

  Full weight: (output_size, input_size)
  GPU 0 holds: (output_size, input_size/tp_size)   ← first chunk of columns
  GPU 1 holds: (output_size, input_size/tp_size)   ← second chunk of columns
'''
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1) # tp_dim = 1 means we will split the weight matrix along the input dimension (columns)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # Bias only added by rank 0 to avoid adding it tp_size times after all-reduce.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y) # each GPU computes a partial output using its shard of the weight matrix, then we sum across GPUs to get the final output. This is necessary because each GPU only sees part of the input features, so their outputs are partial contributions to the final result.
        return y
