"""Prepare native OPD / ATI / FAD configs. Does not launch Ray or training."""
import argparse
import hashlib
import json
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
RECOVERY = LOCAL / "runs/paper4gpu_recovery_20261119T203843Z"
BASES = {
    "deepseek": RECOVERY / "deepseek15-vanilla.json",
    "nemotron": LOCAL / "runs/paper4gpu_20261125T093352Z/nemotron15-opd.json",
}
MODES = ("opd", "ati-opd", "fad-opd", "fixed", "fad-rsi")
LOCATOR_KEYS = ("student_segment_length", "gap_metric", "gap_threshold", "gap_aggregation",
                "max_interventions_per_sample", "teacher_temperature", "teacher_top_p", "teacher_top_k")


def profile(pair, mode, tag):
    base = json.loads(BASES[pair].read_text())
    ati = json.loads((RECOVERY/f"{pair}15-ati.json").read_text())
    if mode == "ati-opd":
        base.update({k:v for k,v in ati.items() if k.startswith("distillation.intervention.") or
                     k.startswith("distillation.diagnostics.")})
    elif mode != "opd":
        for field in LOCATOR_KEYS:
            key = "distillation.intervention."+field
            base[key] = ati[key]
        base["+trainer.fad"] = dict(mode=mode, beta=0.1, opd_weight=1., feedback_tokens=1024, revision_reserve=8192)
    run = f"{pair}-{mode}-{tag}"
    output = Path("/data/fd/donghaosen/fad-rsi/native")/run
    short = hashlib.sha256(run.encode()).hexdigest()[:12]
    base.update({"trainer.experiment_name":run, "trainer.default_local_dir":str(output/"checkpoints"),
                 "trainer.test_data_dir":str(output/"tests"), "trainer.metric_logging.archive.local_dir":str(output/"metric_archive"),
                 "+ray_kwargs.ray_init._temp_dir":f"/data/fd/donghaosen/fr{short}/ray"})
    return base


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="comparison-v1")
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c in "-_" for c in args.tag):
        p.error("tag must be alphanumeric with optional '-' or '_'")
    folder = LOCAL/"fad_configs"/args.tag
    folder.mkdir(parents=True,exist_ok=True)
    for pair in BASES:
        for mode in MODES:
            target=folder/f"{pair}-{mode}.json"
            target.write_text(json.dumps(profile(pair,mode,args.tag),indent=2)+"\n")
    print(folder)
