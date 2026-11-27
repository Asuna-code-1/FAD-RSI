"""Training-only feedback and revision on the existing vLLM and reward workers."""
import json
from uuid import uuid4

import torch

from verl.trainer.distillation.fad_memory import retrieve

FIELDS = ("error", "type", "reason", "strategy", "action", "verification", "conditions")


def diagnosis(text):
    decoder = json.JSONDecoder()
    text = text.rsplit("</think>", 1)[-1]
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if all(isinstance(data.get(k), str) and data[k].strip() for k in FIELDS):
            return {k: data[k].strip() for k in FIELDS}
    return None


async def repair(worker, agent_loop, output, args, sampling, memory):
    from .agent_loop import (
        _compute_intervention_gap, _extract_outcome_correctness, _response_relative_nonfinite_positions,
        _select_intervention_segment, _teacher_gap_logprobs, _teacher_response_logprobs,
    )
    cfg = worker.config.trainer.fad
    gate = worker.teacher_intervention_config
    await worker._compute_score([output], kwargs=args)
    passed, _ = _extract_outcome_correctness(output, gate)
    ids = output.response_ids
    tok = worker.tokenizer
    decode = lambda seq: tok.decode(seq, skip_special_tokens=True)
    draft = decode(ids)
    task = str(args["extra_info"]["index"])
    e = dict(task=task, passed=passed, draft=draft, ok=False, valid=len(ids) if passed else 0,
             attempts=[], chosen=[], prior=None, prior_ok=False, transfer=False,
             teacher_input_tokens=0, teacher_tokens=0, revision_tokens=0, location_tokens=0)
    if passed or not ids:
        return e
    route = worker._teacher_routing_key(args)
    score = await worker._score_sequence_for_intervention(request_id=uuid4().hex, prompt_ids=output.prompt_ids,
        response_ids=ids, output=output, routing_key=route, prompt_logprobs=0)
    t = _teacher_response_logprobs(_teacher_gap_logprobs(score), len(output.prompt_ids), len(ids))
    s = torch.tensor(output.response_logprobs, dtype=torch.float32)
    if t.shape != s.shape:
        raise ValueError("FAD sampled token log-probabilities are misaligned")
    eligible = torch.isfinite(t) & torch.isfinite(s) & (t <= 1e-6) & (s <= 1e-6)
    invalid = _response_relative_nonfinite_positions(score, prompt_len=len(output.prompt_ids), response_len=len(ids))
    eligible[invalid] = False
    gap = torch.where(eligible, _compute_intervention_gap(s,t,gate.gap_metric), 0.)
    span = _select_intervention_segment(gap, eligible.int().tolist(), gate)
    focus = decode(ids[span[0]:span[1]]) if span else ""
    e.update(k3_segment=span, location_tokens=len(ids))
    raw = args["raw_prompt"]
    key = retrieve(memory, raw[-1]["content"]+" "+(focus or draft[-1500:])) if memory else None
    e["prior"] = key
    limit = int(worker.rollout_config.max_model_len)
    cap = int(worker.rollout_config.response_length)

    async def context(instruction, reserve):
        excerpt = ids
        while True:
            messages = [dict(m) for m in raw]
            messages[-1]["content"] += "\nOriginal attempt:\n"+decode(excerpt)+"\n"+instruction
            prompt = await agent_loop.apply_chat_template(messages)
            if len(prompt)+reserve <= limit:
                return prompt
            if not excerpt:
                raise ValueError("FAD question and feedback exceed the context budget")
            keep = max(0, len(excerpt) - (len(prompt)+reserve-limit) - 8)
            excerpt = excerpt[:keep//2] + (excerpt[-(keep-keep//2):] if keep else [])

    async def revise(hint):
        prompt = await context("Solve the original question again using this feedback:\n"+hint, min(cap,cfg.revision_reserve))
        result = await worker.llm_client.generate(request_id=uuid4().hex, prompt_ids=prompt, usage_kind="student_suffix",
            sampling_params={**sampling, "max_tokens":min(cap,limit-len(prompt)), "logprobs":None})
        revised = list(result.token_ids or [])
        verified = False
        if revised:
            candidate = output.model_copy(update=dict(response_ids=revised, response_mask=[1]*len(revised),
                response_logprobs=None, reward_score=None, extra_fields={}))
            await worker._compute_score([candidate], kwargs=args)
            verified, _ = _extract_outcome_correctness(candidate, gate)
        e["attempts"].append(dict(text=decode(revised), verified=verified))
        e["revision_tokens"] += len(revised)
        if verified:
            e.update(ok=True, chosen=revised)
        return verified

    prior = memory[key]["instruction"] if key else ""
    if key and await revise(prior):
        e.update(prior_ok=True, transfer=bool(memory[key]["tasks"]) and task not in memory[key]["tasks"])
        return e
    request = ("Diagnose the earliest error. Return JSON with string fields "+", ".join(FIELDS)+
               ". Quote the incorrect span verbatim in error. Give a transferable hint, no solution or final answer.")
    if focus:
        request += "\nInspect this candidate in context: "+focus
    prompt = await context(request, cfg.feedback_tokens)
    answer = await worker.teacher_server_manager.generate_teacher_segment_single(prefix_ids=prompt,
        max_tokens=cfg.feedback_tokens, routing_key=route, request_id=uuid4().hex)
    feedback = decode(answer.token_ids)
    d = diagnosis(feedback)
    e.update(feedback=feedback, diagnosis=d, teacher_tokens=len(answer.token_ids), teacher_input_tokens=len(prompt))
    if d:
        if draft.count(d["error"]) == 1:
            for a,b in zip(tok.encode(draft.split(d["error"])[0], add_special_tokens=False), ids):
                if a != b:
                    break
                e["valid"] += 1
        await revise(prior+" "+json.dumps(d,ensure_ascii=False))
    return e
