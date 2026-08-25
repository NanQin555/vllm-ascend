import torch

from vllm_ascend.ops.shmem_runtime import (
    _build_weight_for_shmem,
    _share_weight_storage_with_shmem,
)


def test_shmem_weight_reuses_parameter_storage():
    layer = torch.nn.Linear(3, 5, bias=False)
    expected = layer.weight.detach().clone()

    weight_t = _build_weight_for_shmem(layer)
    _share_weight_storage_with_shmem(layer, weight_t)

    assert tuple(layer.weight.shape) == (5, 3)
    assert tuple(weight_t.shape) == (3, 5)
    assert weight_t.is_contiguous()
    assert torch.equal(layer.weight, expected)
    assert (
        layer.weight.untyped_storage().data_ptr()
        == weight_t.untyped_storage().data_ptr()
    )
    assert _build_weight_for_shmem(layer).data_ptr() == weight_t.data_ptr()
