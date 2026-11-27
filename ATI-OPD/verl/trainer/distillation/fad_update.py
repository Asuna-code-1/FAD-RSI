"""Construct actor-only preference rows, leaving the rollout batch intact for baseline metrics."""
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from verl import DataProto
from verl.trainer.distillation.fad_memory import evolve


def actor_batch(trainer, batch, events):
    cfg = trainer.config
    n = len(batch)
    if (cfg.actor_rollout_ref.actor.ppo_epochs != 1 or
        cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n != n):
        raise ValueError("FAD split-pair gradient requires one optimizer update over the complete rollout batch")
    fields = ["input_ids", "position_ids", "attention_mask", "responses", "response_mask", "prompts"]
    original = batch.select(batch_keys=fields + ["old_log_probs", "advantages", "student_topk_ids"],
                            non_tensor_batch_keys=[], meta_info_keys=["temperature"])
    responses = batch.batch["responses"]
    cutoff = torch.tensor([e["valid"] for e in events])
    response_positions = torch.arange(responses.size(1), device=responses.device)
    original.batch["policy_loss_mask"] = (batch.batch["response_mask"].bool() &
        (response_positions[None, :] < cutoff[:, None]))
    # Zero-supervision rows use one real token to keep FSDP packing and rank sizes uniform.
    preferred = batch.select(batch_keys=fields, non_tensor_batch_keys=[], meta_info_keys=["temperature"])
    preferred.batch = preferred.batch.clone()
    p = preferred.batch
    p["responses"].fill_(trainer.tokenizer.pad_token_id)
    p["response_mask"].zero_()
    for i,e in enumerate(events):
        tokens = e["chosen"] or [trainer.tokenizer.eos_token_id]
        p["responses"][i,:len(tokens)] = torch.tensor(tokens)
        p["response_mask"][i,:len(tokens)] = 1
    width = p["prompts"].size(1)
    p["input_ids"][:,width:] = p["responses"]
    p["attention_mask"][:,width:] = p["response_mask"]
    p["position_ids"] = (p["attention_mask"].long().cumsum(-1)-1).clamp_min(0)*p["attention_mask"]
    weights = torch.zeros(n)
    dpo_value = torch.zeros(n)
    pos_old = torch.zeros(n)
    neg_old = torch.zeros(n)
    paired = torch.tensor([bool(e["chosen"]) for e in events])
    dpo_start = torch.tensor([e["dpo_start"] if e["chosen"] else responses.size(1) for e in events])
    if paired.any():
        def mean(lp, mask):
            return (lp.masked_fill(~mask.bool(),0).sum(-1)/mask.sum(-1).clamp_min(1))
        # Both continuations are scored under the original question and preserved prefix.
        suffix_positions = response_positions[None, :] >= dpo_start[:, None]
        neg_mask = batch.batch["response_mask"].bool() & suffix_positions
        pos_mask = p["response_mask"].bool() & suffix_positions
        neg_old = mean(batch.batch["fad_sample_logp"], neg_mask)
        pos_old = mean(trainer._compute_ref_log_prob(preferred, current_actor=True).batch["ref_log_prob"],pos_mask)
        pos_ref = mean(trainer._compute_ref_log_prob(preferred).batch["ref_log_prob"],pos_mask)
        neg_ref = mean(batch.batch["ref_log_prob"],neg_mask)
        beta = cfg.trainer.fad.beta
        margin = beta*((pos_old-neg_old)-(pos_ref-neg_ref))
        weights[paired] = -beta*torch.sigmoid(-margin[paired])/paired.sum()
        dpo_value[paired] = F.softplus(-margin[paired])/paired.sum()
    for key in ("student_topk_ids","old_log_probs","advantages","policy_loss_mask"):
        p[key] = torch.zeros_like(original.batch[key])
    for proto, slope, old, value in [(original,-weights,neg_old,dpo_value),(preferred,weights,pos_old,torch.zeros(n))]:
        proto.batch["fad_slope"] = slope
        proto.batch["fad_old_mean"] = old
        proto.batch["fad_value"] = value
        proto.batch["fad_dpo_start"] = dpo_start
    result = DataProto.concat([original, preferred])
    # Keep the original DP balance when the batch is split across student ranks.
    indices = torch.arange(n)
    result.reorder(torch.stack((indices, indices + n), dim=1).flatten())
    return result


def record(trainer, events, metrics):
    step = trainer.global_steps
    folder = Path(trainer.config.trainer.default_local_dir).parent / "fad_experiences"
    folder.mkdir(parents=True,exist_ok=True)
    tmp = folder/f"step-{step}.tmp"
    with tmp.open("w") as stream:
        for e in events:
            stream.write(json.dumps(e,ensure_ascii=False)+"\n")
    os.replace(tmp, folder/f"step-{step}.jsonl")
    if trainer.config.trainer.fad.mode == "fad-rsi":
        for e in events:
            evolve(trainer.fad_memory,e)
    errors = sum(not e["passed"] for e in events)
    ok = sum(e["ok"] for e in events)
    metrics.update({"fad/preference_pairs":ok, "fad/error_recovery_rate":ok/max(errors,1),
        "fad/revision_success_rate":ok/max(sum(len(e["attempts"]) for e in events),1),
        "fad/cross_task_repairs":sum(e["transfer"] for e in events),
        "fad/prior_success_rate":sum(e["prior_ok"] for e in events)/max(sum(bool(e["prior"]) for e in events),1),
        "fad/teacher_feedback_calls":sum("feedback" in e for e in events),
        **{f"fad/{k}":sum(e[k] for e in events) for k in
           ("teacher_input_tokens","teacher_tokens","revision_tokens","location_tokens")}})
