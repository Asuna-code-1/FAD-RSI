import json
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "ATI-OPD" / "local/runs"
PROFILES = {
    "deepseek": BASE / "paper4gpu_recovery_20261119T203843Z/deepseek15-vanilla.json",
    "nemotron": BASE / "paper4gpu_20261125T093352Z/nemotron15-opd.json",
}
MODES = ("opd", "fad-opd", "fixed", "fad-rsi")


def load(pair):
    cfg = json.loads(PROFILES[pair].read_text())
    # Vanilla defines the training comparison. ATI supplies its enabled locator settings.
    ati_path = BASE / "paper4gpu_recovery_20261119T203843Z" / f"{pair}15-ati.resolved.yaml"
    ati = yaml.safe_load(ati_path.read_text())
    settings = ati["distillation"]["intervention"]
    cfg["fad.locator_source"] = str(ati_path)
    cfg["fad.locator"] = {k: settings[k] for k in (
        "gap_metric", "gap_aggregation", "gap_threshold", "student_segment_length",
        "teacher_rejection_persistence_enabled", "max_interventions_per_sample",
        "teacher_temperature", "teacher_top_p", "teacher_top_k")}
    if (settings["gap_metric"] != "k3" or settings["max_interventions_per_sample"] != 1
            or settings["teacher_rejection_persistence_enabled"]):
        raise ValueError("Expected the historical single-segment K3 configuration")
    actor = ati["actor_rollout_ref"]["actor"]
    for name in ("betas", "weight_decay"):
        cfg.setdefault("actor_rollout_ref.actor.optim." + name, actor["optim"][name])
    cfg.setdefault("actor_rollout_ref.actor.grad_clip", actor["grad_clip"])
    for path in [*cfg["data.train_files"], *cfg["data.val_files"],
                 cfg["actor_rollout_ref.model.path"], cfg["distillation.teacher_models.teacher_model.model_path"]]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    return cfg
