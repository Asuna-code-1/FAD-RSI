import pytest
import torch

from verl.utils.chunked_logprobs import selected_logprobs


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('shape', [(19, 37), (2, 11, 37)])
def test_values_and_gradients_match_dense(dtype, shape):
    torch.manual_seed(7)
    logits = torch.randn(shape, dtype=dtype)
    logits[..., -1] = -float('inf')  # Tokenizer-masked vocabulary entries.
    logits.requires_grad_()
    ids = torch.randint(0, 36, (*shape[:-1], 5))
    ids[..., 1] = ids[..., 0]  # Duplicate targets must accumulate gradients.
    expected = logits.float().gather(-1, ids) - logits.float().logsumexp(-1, keepdim=True)
    actual = selected_logprobs(logits, ids, chunk_size=7)
    weight = torch.randn_like(actual)
    expected_grad, = torch.autograd.grad((expected * weight).sum(), logits)
    actual_grad, = torch.autograd.grad((actual * weight).sum(), logits)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    tolerance = 0.008 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual_grad, expected_grad, rtol=tolerance, atol=tolerance)


def test_backward_does_not_save_full_float32_logits():
    logits = torch.randn(19, 37, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.randint(0, 37, (19, 5))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: (saved.append(x), x)[1], lambda x: x):
        selected_logprobs(logits, ids, chunk_size=7).sum().backward()
    assert not any(t.shape == logits.shape and t.dtype == torch.float32 for t in saved)


def test_diagnostic_topk_matches_original_normalization():
    from verl.utils.torch_functional import topk_logprobs_from_logits

    torch.manual_seed(9)
    logits = torch.randn(2, 29, 67, dtype=torch.bfloat16)
    values, ids = logits.topk(16, dim=-1)
    expected = values.float() - logits.float().logsumexp(-1, keepdim=True)
    actual_ids, actual = topk_logprobs_from_logits(logits, 16)
    torch.testing.assert_close(actual_ids, ids)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
