import json
import re
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from verl.utils.reward_score import default_compute_score
from strategy import retrieve

FIELDS = ("error", "type", "reason", "strategy", "action", "verification", "conditions")


def correct(row, answer):
    result = default_compute_score(row["data_source"], answer, row["reward_model"]["ground_truth"], row["extra_info"])
    return bool(result["acc"]) if isinstance(result, dict) and "acc" in result else float(
        result["score"] if isinstance(result, dict) else result) > 0


class Repair:
    def __init__(self, student, teacher, tok, cfg):
        self.student, self.teacher, self.tok, self.cfg = student, teacher, tok, cfg
        self.limit = cfg["actor_rollout_ref.rollout.max_model_len"]

    def prompt(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)

    @torch.no_grad()
    def generate(self, model, prompt, cap, temperature=None, top_p=None, top_k=None):
        cfg = self.cfg
        ids = torch.tensor([prompt], device=model.device)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                             temperature=temperature or cfg["actor_rollout_ref.rollout.temperature"],
                             top_p=top_p or cfg["actor_rollout_ref.rollout.top_p"], top_k=max(0, cfg["actor_rollout_ref.rollout.top_k"] if top_k is None else top_k),
                             max_new_tokens=min(cap, self.limit-len(prompt)), use_cache=True,
                             pad_token_id=self.tok.eos_token_id)
        return out[0, len(prompt):].tolist()

    def context(self, row, draft, instruction, reserve):
        # Keep the entire question and instruction. Retain both ends of a long draft.
        ids = self.tok.encode(draft, add_special_tokens=False)
        budget = self.limit - reserve
        while True:
            excerpt = self.tok.decode(ids, skip_special_tokens=True)
            messages = [dict(m) for m in row["prompt"]]
            messages[-1]["content"] += "\nOriginal attempt:\n" + excerpt + "\n" + instruction
            prompt = self.prompt(messages)
            if len(prompt) <= budget:
                return prompt
            keep = max(0, len(ids) - max(16, len(prompt)-budget+8))
            if not ids:
                raise ValueError("Question and instruction exceed the context budget")
            ids = ids[:keep//2] + (ids[-(keep-keep//2):] if keep else [])

    @torch.no_grad()
    def locate(self, prompt, response):
        from verl.experimental.agent_loop.agent_loop import _compute_intervention_gap, _select_intervention_segment

        def sampled_logps(model):
            states = hidden(model, prompt, response)
            ids = torch.tensor(response, device=states.device)
            return torch.cat([
                -F.cross_entropy(model.lm_head(part).float(), target, reduction="none").cpu()
                for part, target in zip(states.split(128), ids.split(128))
            ])

        settings = SimpleNamespace(**self.cfg["fad.locator"])
        gaps = _compute_intervention_gap(sampled_logps(self.student), sampled_logps(self.teacher), settings.gap_metric)
        return _select_intervention_segment(gaps, [1] * len(response), settings)

    def sample(self, row, memory, mode):
        prompt = self.prompt(row["prompt"])
        cap = self.cfg["data.max_response_length"]
        original = self.generate(self.student, prompt, cap)
        decode = lambda ids: self.tok.decode(ids, skip_special_tokens=True)
        draft = decode(original)
        passed = correct(row, draft)
        e = dict(task=str(row["extra_info"]["index"]), draft=draft, passed=passed, ok=False,
                 teacher_tokens=0, teacher_input_tokens=0, revision_tokens=0, attempts=[])
        valid = len(original) if passed or mode == "opd" else 0
        chosen = None
        def revise(feedback):
            p = self.context(row, draft, "Solve the original question again. Use this feedback: "+feedback, min(cap, 8192))
            ids = self.generate(self.student, p, cap)
            answer = decode(ids)
            ok = correct(row, answer)
            e["revision_tokens"] += len(ids)
            e["attempts"].append(dict(text=answer, verified=ok))
            return ids if ok else None
        if not passed and mode != "opd":
            segment = self.locate(prompt, original)
            e["teacher_localization_tokens"] = len(original)
            e["k3_segment"] = None
            focus = ""
            if segment is not None:
                start, end, score = segment
                focus = decode(original[start:end])
                e["k3_segment"] = dict(start=start, end=end, score=score, text=focus)
            key = retrieve(memory, row["prompt"][-1]["content"]+" "+(focus or draft[-1500:])) if mode != "fad-opd" else None
            e["prior"] = key
            if key:
                chosen = revise(memory[key]["instruction"])
                e["prior_ok"] = chosen is not None
                e["transfer"] = e["prior_ok"] and bool(memory[key]["tasks"]) and e["task"] not in memory[key]["tasks"]
            if chosen is None:
                request = ("Diagnose the first error. Return a JSON object with string fields "+", ".join(FIELDS)+
                           ". error must quote the first incorrect span verbatim. conditions and strategy must "
                           "generalize to new problems. Give a short actionable hint, no final answer or worked solution.")
                if focus:
                    request += "\nInspect this candidate span in context and verify whether an earlier error exists:\n" + focus
                p = self.context(row, draft, request, 1024)
                settings = self.cfg["fad.locator"]
                feedback = self.generate(self.teacher, p, 1024, settings["teacher_temperature"],
                                         settings["teacher_top_p"], settings["teacher_top_k"])
                e.update(teacher_tokens=len(feedback), teacher_input_tokens=len(p), feedback=decode(feedback))
                content = e["feedback"].split("</think>")[-1]
                decoder = json.JSONDecoder()
                for start in reversed([m.start() for m in re.finditer(r"\{", content)]):
                    try:
                        d, _ = decoder.raw_decode(content[start:])
                    except ValueError:
                        continue
                    if all(isinstance(d.get(k), str) and d[k].strip() for k in FIELDS):
                        e["diagnosis"] = {k: d[k].strip() for k in FIELDS}
                        break
                if d := e.get("diagnosis"):
                    if draft.count(d["error"]) == 1:
                        prefix = self.tok.encode(draft.split(d["error"])[0], add_special_tokens=False)
                        for a,b in zip(prefix,original):
                            if a != b:
                                break
                            valid += 1
                    prior = memory[key]["instruction"] if key else ""
                    chosen = revise(prior+" "+json.dumps(d, ensure_ascii=False))
            e["ok"] = chosen is not None
        return (prompt, original, valid, chosen), e


def hidden(model, prompt, response):
    ids = torch.tensor([prompt+response], device=model.device)
    return model.model(ids, use_cache=False).last_hidden_state[0,len(prompt)-1:-1]


def mean_logp(model, prompt, response):
    h = hidden(model,prompt,response)
    ids = torch.tensor(response, device=h.device)
    def selected(part, target):
        return -F.cross_entropy(model.lm_head(part).float(), target, reduction="sum")
    return sum(checkpoint(selected, part, target, use_reentrant=False)
               for part,target in zip(h.split(128),ids.split(128))) / len(response)


def opd_sum(student, teacher, prompt, response, cfg):
    h = hidden(student,prompt,response)
    with torch.no_grad():
        t = hidden(teacher,prompt,response)
    k = cfg["distillation.distillation_loss.topk"]
    floor = cfg["distillation.distillation_loss.log_prob_min_clamp"]
    def token_loss(part, other):
        lp = student.lm_head(part).float().log_softmax(-1)
        values, indices = lp.topk(k)
        with torch.no_grad():
            target = teacher.lm_head(other).float().log_softmax(-1).gather(-1,indices.to(other.device)).to(lp.device)
            old = values.detach().clamp_min(floor)
            reward = old.softmax(-1) * (target.clamp_min(floor)-old)
        # On-policy PG evaluated at rollout weights, one update per collected batch.
        return -(reward * (values-values.detach()).exp()).sum()
    return sum(checkpoint(token_loss,a,b,use_reentrant=False) for a,b in zip(h.split(128),t.split(128)))


def preference(student, ref, prompt, chosen, rejected, beta):
    with torch.no_grad():
        reference = (mean_logp(ref,prompt,chosen)-mean_logp(ref,prompt,rejected)).to(student.device)
    margin = mean_logp(student,prompt,chosen)-mean_logp(student,prompt,rejected)
    return -F.logsigmoid(beta*(margin-reference))
