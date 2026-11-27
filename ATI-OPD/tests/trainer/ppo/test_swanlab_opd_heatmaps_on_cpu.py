"""SwanLab-only logging must retain upstream OPD heatmaps."""
from types import SimpleNamespace

import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_swanlab_only_heatmaps_are_generated_without_wandb_run():
    trainer = object.__new__(RayPPOTrainer)
    trainer.logger = SimpleNamespace(logger={'swanlab': object()})
    trainer.distillation_config = SimpleNamespace(diagnostics=SimpleNamespace(
        enabled=True, log_wandb_heatmap=True, heatmap_interval=10,
        heatmap_history_steps=80, position_bin_size=2, high_gap_threshold=1.0,
    ))
    trainer.opd_entropy_tracker = None
    trainer.opd_gap_tracker = None
    trainer.opd_gap_high_tracker = None
    trainer.opd_teacher_logprob_nonfinite_tracker = None
    metrics = trainer._maybe_log_opd_heatmaps(
        global_steps=10,
        entropy=torch.ones(1, 6), student_logprobs=-torch.ones(1, 6),
        teacher_logprobs=-torch.ones(1, 6)*2,
        teacher_logprob_nonfinite_positions=None, response_mask=torch.ones(1, 6),
    )
    assert set(metrics) == {'opd/entropy_heatmap', 'opd/gap_abs_heatmap', 'opd/gap_abs_high_heatmap'}
    assert all(type(value).__module__.startswith('swanlab.') for value in metrics.values())
