"""Build from every upstream launcher override, with explicit hardware/user exceptions."""
import datetime
import json
from pathlib import Path

ROOT = Path('/workspace/fudan/donghaosen/ATI-OPD')
active = ROOT/'local/active-four-gpu-run.json'
previous_path = Path(json.loads(active.read_text())['manifest'])
previous = json.loads(previous_path.read_text())
assert previous['status'] in ('stopped_by_user', 'failed_queue_paused', 'stopped_memory_limit')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
batch = 'paper4gpu_aligned_' + stamp
code = ROOT/'local/runs'/batch
data = Path('/data/fd/donghaosen/ati-opd/runs')/batch
code.mkdir(parents=True)
data.mkdir(parents=True)
manifest = {k: v for k, v in previous.items() if k not in ('runs', 'updated_at_utc', 'previous_attempt', 'retry_reason')}
manifest.update(batch=batch, data_root=str(data), status='prepared', previous_attempt=str(previous_path),
                created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                retry_reason='User-requested full rerun after bounded-logprob memory fix; target 120 GiB',
                memory_stop_mib_per_gpu=138240, runs=[], notes=[
                    'GPU 0-3: 2 student + 2 teacher; about 120 GiB target, 135 GiB emergency stop.',
                    'Fresh weights and measured step0, shuffle=False, upstream algorithm and full diagnostics.',
                    '512-token FP32 logprob workspace; student inference memory fraction 0.55 for headroom.',
                    'ATI DeepSeek then ATI Nemotron, followed by both vanilla OPD baselines.',
                ])
for index, old in enumerate(previous['runs']):
    variant = old['variant']
    original = json.loads(Path(old['config']).read_text())
    method = 'ati' if variant.endswith('-ati') else 'vanilla'
    upstream = json.loads((ROOT/f'local/upstream-{method}-overrides.json').read_text())
    cfg = dict(upstream)
    out = data/variant
    for folder in ('logs', 'checkpoints', 'rollouts', 'tests', 'swanlab', 'metric_archive'):
        (out/folder).mkdir(parents=True)
    # Ray appends its session and socket names; keep the UNIX socket path short.
    short = Path('/data/fd/donghaosen')/('a'+stamp[9:15]+str(index))
    short.mkdir(parents=True)
    assert len(str(short/'ray/session_2026-11-18_22-07-01_276824_3098546/sockets/plasma_store')) < 100
    # Paths are mapped to verified, already-downloaded assets.
    paths = {key: original[key] for key in (
        'data.train_files', 'data.val_files', 'data.test_benchmarks.root',
        'actor_rollout_ref.model.path', 'actor_rollout_ref.model.chat_template_from_model',
        'distillation.teacher_models.teacher_model.model_path',
    )}
    cfg.update(paths)
    hardware = {
        'trainer.device': 'cuda', 'trainer.n_gpus_per_node': 2, 'trainer.nnodes': 1,
        'distillation.n_gpus_per_node': 2, 'distillation.nnodes': 1,
        'distillation.teacher_models.teacher_model.num_replicas': 2,
        'actor_rollout_ref.actor.strategy': 'fsdp2',
        'actor_rollout_ref.actor.ulysses_sequence_parallel_size': 1,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size': 1,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': 24704,
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': 24704,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.enforce_eager': True,
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.55,
        'distillation.teacher_models.teacher_model.inference.enforce_eager': True,
        'data.dataloader_num_workers': 4,
    }
    hardware.update({key: value for key, value in original.items() if 'ray_kwargs.' in key})
    hardware['+ray_kwargs.ray_init._temp_dir'] = str(short/'ray')
    cfg.update(hardware)
    user_settings = {
        'trainer.logger': ['console', 'swanlab'],
        'trainer.project_name': 'ati-opd',
        'trainer.experiment_name': variant+'-aligned-step0-'+stamp,
        'trainer.default_local_dir': str(out/'checkpoints'),
        'trainer.test_data_dir': str(out/'tests'),
        'trainer.metric_logging.archive.local_dir': str(out/'metric_archive'),
        'trainer.metric_logging.archive.upload_to_wandb': 'disabled',
        'trainer.test_before_train': True,
        'data.test_benchmarks.names': original['data.test_benchmarks.names'],
        'trainer.total_training_steps': 217,
        'trainer.resume_mode': 'disable',
        'trainer.del_local_ckpt_after_load': False,
        'trainer.max_actor_ckpt_to_keep': None,
        'trainer.max_critic_ckpt_to_keep': None,
    }
    cfg.update(user_settings)
    assert cfg['data.shuffle'] is False and cfg['data.seed'] == -42
    assert cfg['distillation.diagnostics.enabled'] is True
    assert cfg['trainer.total_epochs'] == 1 and cfg['trainer.test_before_train'] is True
    assert cfg['actor_rollout_ref.actor.use_torch_compile'] is False
    assert cfg['actor_rollout_ref.rollout.gpu_memory_utilization'] == 0.55
    assert cfg['actor_rollout_ref.rollout.max_num_batched_tokens'] == 2048
    allowed = set(paths) | set(hardware) | set(user_settings)
    differences = {key: {'upstream': value, 'effective': cfg[key]} for key, value in upstream.items() if value != cfg[key]}
    assert set(differences).issubset(allowed)
    path = code/(variant+'.json')
    path.write_text(json.dumps(cfg, indent=2)+'\n')
    (code/(variant+'.alignment.json')).write_text(json.dumps({
        'upstream_commit': '10db7986937784dffee5597cae7778770f81c471',
        'upstream_override_count': len(upstream), 'unexplained_differences': [],
        'documented_differences': differences,
        'exception_reasons': {'paths': 'Verified local datasets and weights',
            'hardware': 'Four H200 GPUs, 2 student + 2 teacher, CUDA/FSDP2 and compatible Ray runtime',
            'user_settings': 'SwanLab only, fresh run, step-0 evaluation, five paper benchmarks, retain artifacts'},
    }, indent=2)+'\n')
    manifest['runs'].append({'variant': variant, 'tag': variant+'-aligned-'+stamp, 'config': str(path),
        'output_dir': str(out), 'short_tmp_dir': str(short), 'session': 'ati-'+variant+'-aligned-'+stamp,
        'logging_backend': 'swanlab', 'state': 'prepared'})
path = code/'manifest.json'
path.write_text(json.dumps(manifest, indent=2)+'\n')
(ROOT/'local/prepared-aligned-campaign.json').write_text(json.dumps({'manifest': str(path)}, indent=2)+'\n')
print(path)
