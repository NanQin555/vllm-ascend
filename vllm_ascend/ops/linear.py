# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
To customize linear communication groups or forward of classes in this file,
extend new linear operations in linear_op.py.
The classes in this file should not be modified, including AscendQKVParallelLinear,
AscendMergedColumnParallelLinear, AscendMergedColumnParallelLinear,
AscendRowParallelLinear and AscendColumnParallelLinear.
"""
import os

from typing import Optional, Union
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from vllm.config import get_current_vllm_config
from vllm.distributed import divide
from vllm.model_executor.layers.linear import (  # noqa
    WEIGHT_LOADER_V2_SUPPORTED, ColumnParallelLinear, LinearBase,
    MergedColumnParallelLinear, QKVParallelLinear, QuantizeMethodBase,
    ReplicatedLinear, RowParallelLinear, UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.distributed import tensor_model_parallel_all_reduce

from vllm_ascend.ops.linear_op import get_parallel_op, get_replicated_op
from vllm_ascend.ops.shmem_runtime import (finalize_shmem_matmul_allreduce,
                                           prepare_shmem_matmul_allreduce)
from vllm_ascend.utils import enable_sp, maybe_trans_nz

_SHMEM_ENABLED = (
    os.getenv("VLLM_ASCEND_ENABLE_SHMEM_MATMUL_ALLREDUCE")
    or os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0")
).lower() in {"1", "true", "yes", "on"}


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class AscendUnquantizedLinearMethod(UnquantizedLinearMethod):
    """Linear method without quantization"""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if "conv1d" not in layer.prefix:
            layer.weight.data = maybe_trans_nz(layer.weight.data)
        finalize_shmem_matmul_allreduce(layer)


# TODO(realliujiaxu): Remove this class after linear of vllm supports custom comm group
class AscendLinearBase(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        nn.Module.__init__(self)

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        if quant_config is None:
            self.quant_method: Optional[
                QuantizeMethodBase] = AscendUnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self,
                                                              prefix=prefix)
        self.return_bias = return_bias
        self.disable_tp = disable_tp


class AscendQKVParallelLinear(QKVParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.custom_op, _, tp_size = get_parallel_op(disable_tp, prefix, self,
                                                     "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after linear of vllm supports custom comm group
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size,
                                               self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads +
                       2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]
        AscendColumnParallelLinear.__init__(self,
                                            input_size=input_size,
                                            output_size=output_size,
                                            bias=bias,
                                            gather_output=False,
                                            skip_bias_add=skip_bias_add,
                                            params_dtype=params_dtype,
                                            quant_config=quant_config,
                                            prefix=prefix,
                                            return_bias=return_bias,
                                            disable_tp=disable_tp)

    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(
            disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after linear of vllm supports custom comm group
        self.output_sizes = output_sizes
        assert all(output_size % self.tp_size == 0
                   for output_size in output_sizes)
        AscendColumnParallelLinear.__init__(self,
                                            input_size=input_size,
                                            output_size=sum(output_sizes),
                                            bias=bias,
                                            gather_output=gather_output,
                                            skip_bias_add=skip_bias_add,
                                            params_dtype=params_dtype,
                                            quant_config=quant_config,
                                            prefix=prefix,
                                            return_bias=return_bias,
                                            disable_tp=disable_tp)

    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendRowParallelLinear(RowParallelLinear):
    """Linear layer with row parallelism.
    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    # NOTE: Globally unique prefix identifier used in SP scenarios
    unique_prefix_idx = 0

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # TODO(kunpengW-code): Specifying the prefix in linear layers of some models in the vLLM.
        self.unique_prefix = prefix
        if enable_sp() or _SHMEM_ENABLED:
            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                self.unique_prefix = (
                    f"{prefix}.unique_prefix"
                    f"{AscendRowParallelLinear.unique_prefix_idx}"
                )
                AscendRowParallelLinear.unique_prefix_idx += 1
            compilation_config.static_forward_context[self.unique_prefix] = self

        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(
            disable_tp, prefix, self, "row")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after linear of vllm supports custom comm group
        # Divide the weight matrix along the first dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        AscendLinearBase.__init__(self,
                                  input_size,
                                  output_size,
                                  skip_bias_add,
                                  params_dtype,
                                  quant_config,
                                  prefix,
                                  return_bias=return_bias,
                                  disable_tp=disable_tp)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

        self._can_try_shmem_matmul_allreduce = (
            _SHMEM_ENABLED
            and reduce_results
            and self.tp_size > 1
            and any(token in prefix for token in ("o_proj", "down_proj"))
            and "UnquantizedLinearMethod" in type(self.quant_method).__name__
        )
        if self._can_try_shmem_matmul_allreduce:
            prepare_shmem_matmul_allreduce(self)
        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(
        self,
        input_,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        assert self.quant_method is not None

        if self._can_try_shmem_matmul_allreduce:
            output = torch.ops.vllm.shmem_matmul_allreduce(
                input_parallel, self.unique_prefix
            )
        else:
            bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
            output_parallel = self.quant_method.apply(self, input_parallel, bias_)
            if self.reduce_results and self.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output_parallel)
            else:
                output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class AscendColumnParallelLinear(ColumnParallelLinear):
    """Linear layer with column parallelism.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[list[int]] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(
            disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after linear of vllm supports custom comm group
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]

        AscendLinearBase.__init__(self,
                                  input_size,
                                  output_size,
                                  skip_bias_add,
                                  params_dtype,
                                  quant_config,
                                  prefix,
                                  return_bias=return_bias,
                                  disable_tp=disable_tp)

        self.gather_output = gather_output

        if output_sizes is None:
            output_sizes = [output_size]

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


class AscendReplicatedLinear(ReplicatedLinear):
    """Ascend Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: Take no effect for replicated linear layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op = get_replicated_op(disable_tp, prefix, self)
        # If MergedReplicatedLinear, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        AscendLinearBase.__init__(self,
                                  input_size,
                                  output_size,
                                  skip_bias_add,
                                  params_dtype,
                                  quant_config,
                                  prefix=prefix,
                                  return_bias=return_bias,
                                  disable_tp=disable_tp)

        # All the linear layer supports quant method.
        assert self.quant_method is not None
        self.quant_method.create_weights(self,
                                         self.input_size, [self.output_size],
                                         self.input_size,
                                         self.output_size,
                                         self.params_dtype,
                                         weight_loader=self.weight_loader)

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)
