"""Measure dense vs bounded-workspace forward/backward on the full token budget."""
import gc
import json
from pathlib import Path
import torch
from verl.utils.chunked_logprobs import selected_logprobs

torch.manual_seed(7)
results = {}
for mode in ('dense', 'chunked'):
    gc.collect()
    torch.cuda.empty_cache()
    logits = torch.randn((24704, 151936), device='cuda', dtype=torch.bfloat16, requires_grad=True)
    ids = torch.randint(0, 151936, (24704, 16), device='cuda')
    torch.cuda.reset_peak_memory_stats()
    if mode == 'dense':
        values = logits.float()
        result = values.gather(-1, ids) - values.logsumexp(-1, keepdim=True)
    else:
        result = selected_logprobs(logits, ids)
    result.mean().backward()
    torch.cuda.synchronize()
    results[mode] = {'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                     'all_finite': bool(torch.isfinite(result).all() and torch.isfinite(logits.grad).all())}
    print(json.dumps({mode: results[mode]}), flush=True)
    del result, logits, ids
    if mode == 'dense':
        del values
assert results['chunked']['peak_allocated_gib'] < results['dense']['peak_allocated_gib'] * 0.6
assert all(v['all_finite'] for v in results.values())
Path('/data/fd/donghaosen/ati-opd/chunked-logprobs-memory-validation.json').write_text(json.dumps(results, indent=2)+'\n')
