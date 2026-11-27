"""Selected-token log probabilities with bounded float32 workspaces."""
import torch


class _SelectedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, ids, chunk_size):
        ctx.save_for_backward(logits, ids)
        ctx.chunk_size = chunk_size
        result = torch.empty(ids.shape, device=logits.device, dtype=torch.float32)
        for start in range(0, logits.shape[0], chunk_size):
            stop = start + chunk_size
            block = logits[start:stop].float()
            result[start:stop] = block.gather(-1, ids[start:stop]) - torch.logsumexp(block, -1, keepdim=True)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        logits, ids = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)
        for start in range(0, logits.shape[0], ctx.chunk_size):
            stop = start + ctx.chunk_size
            block = logits[start:stop].float()
            upstream = grad_output[start:stop].float()
            grad = torch.softmax(block, dim=-1)
            grad.mul_(-upstream.sum(-1, keepdim=True))
            grad.scatter_add_(-1, ids[start:stop], upstream)
            grad_logits[start:stop] = grad.to(logits.dtype)
        return grad_logits, None, None


def selected_logprobs(logits, ids, chunk_size=512):
    """Same float32 normalization as the dense implementation, including gradients.

    Save the original logits for backward, not a full float32 vocabulary tensor.
    Recompute softmax in token blocks so workspace does not grow with sequence length.
    """
    if chunk_size <= 0 or logits.shape[:-1] != ids.shape[:-1]:
        raise ValueError('Invalid chunk size or token dimensions')
    shape = ids.shape
    return _SelectedLogProbs.apply(
        logits.reshape(-1, logits.shape[-1]),
        ids.reshape(-1, ids.shape[-1]).to(device=logits.device, dtype=torch.long),
        chunk_size,
    ).reshape(shape)
