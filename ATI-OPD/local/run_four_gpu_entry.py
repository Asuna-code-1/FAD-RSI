"""Run a reviewed four-GPU config; credentials are never included in Ray runtime_env."""
import argparse
import json
import os
from pathlib import Path
import sys
import socket
import faulthandler
import signal
faulthandler.register(signal.SIGUSR1, all_threads=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The cloud project now requires SwanLab >=0.9. Keep its protobuf6 runtime
# isolated from the original environment and propagate it to Ray workers.
SDK_RUNTIME = ROOT/'local/runtime-swanlab0103'
sys.path.insert(0, str(SDK_RUNTIME))
os.environ['PYTHONPATH'] = str(SDK_RUNTIME) + os.pathsep + os.environ.get('PYTHONPATH', str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--check-only', action='store_true')
parser.add_argument('--no-write-resolved', action='store_true')
args = parser.parse_args()
settings = json.loads(Path(args.config).read_text())
if '-aligned-' in settings.get('trainer.experiment_name', ''):
    assert settings['data.shuffle'] is False and settings['data.seed'] == -42
    if settings['trainer.resume_mode'] == 'resume_path':
        assert Path(settings['trainer.resume_from_path'], 'metrics_commit.json').is_file()
        assert Path(settings['trainer.resume_from_path'], 'data.pt').is_file()
        assert Path(settings['trainer.resume_from_path'], 'opd_prune_future_calibration.json').is_file()
    else:
        assert settings['trainer.test_before_train'] is True
        assert settings['trainer.resume_mode'] == 'disable'
    assert settings['distillation.diagnostics.enabled'] is True
    assert settings['trainer.total_epochs'] == 1
    assert settings['actor_rollout_ref.actor.use_torch_compile'] is False
    assert settings['trainer.metric_logging.archive.enabled'] is True
def hydra_value(value):
    if isinstance(value, dict):
        return '{' + ','.join(str(k) + ':' + hydra_value(v) for k, v in value.items()) + '}'
    return json.dumps(value, separators=(',', ':'))
overrides = [key + '=' + hydra_value(value) for key, value in settings.items()]
from hydra.core.override_parser.overrides_parser import OverridesParser
OverridesParser.create().parse_overrides(overrides)
if args.check_only:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT/'verl/trainer/config'), version_base=None):
        cfg = compose(config_name='ppo_trainer', overrides=overrides)
    from verl.utils.config import validate_config
    from verl.trainer.ppo.utils import need_reference_policy
    validate_config(cfg, use_reference_policy=need_reference_policy(cfg), use_critic=False)
    from verl.utils.config import omega_conf_to_dataclass
    omega_conf_to_dataclass(cfg.distillation)
    assert cfg.trainer.n_gpus_per_node + cfg.distillation.n_gpus_per_node == 4
    assert cfg.data.train_batch_size == 64 and cfg.actor_rollout_ref.rollout.n == 4
    assert cfg.data.max_response_length == 23552 and cfg.trainer.total_training_steps == 217
    assert cfg.trainer.max_actor_ckpt_to_keep is None and not cfg.trainer.del_local_ckpt_after_load
    for name in cfg.data.train_files + cfg.data.val_files:
        assert Path(name).is_file(), name
    if not args.no_write_resolved:
        OmegaConf.save(cfg, Path(args.config).with_suffix('.resolved.yaml'), resolve=True)
    print(json.dumps({'config':args.config,'status':'composed_and_validated','gpu_split':'2 student + 2 teacher'}))
    raise SystemExit(0)
if not os.environ.get('TMUX') or os.environ.get('ATI_INTERNAL_PORTS_AUTHORIZED') != '1':
    raise SystemExit('This authorized experiment must run in tmux with the internal-port exception.')
loggers = settings['trainer.logger']
assert loggers in (['console', 'swanlab'], ['console', 'swanlab', 'wandb'])
if 'wandb' in loggers:
    if not os.environ.get('WANDB_API_KEY'):
        raise SystemExit('Set WANDB_API_KEY for the selected logger.')
    os.environ['WANDB_MODE'] = 'online'
    logging_root = Path(settings['trainer.default_local_dir']).parent / 'wandb'
    for variable, name in [('WANDB_DIR', '.'), ('WANDB_CACHE_DIR', 'cache'),
                           ('WANDB_DATA_DIR', 'data'), ('WANDB_CONFIG_DIR', 'config')]:
        directory = logging_root / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
else:
    os.environ.pop('WANDB_API_KEY', None)
    os.environ['WANDB_MODE'] = 'disabled'
if not os.environ.get('SWANLAB_API_KEY'):
    raise SystemExit('Set SWANLAB_API_KEY for the selected logger.')
# Avoid Ray's 30-second persisted-port-file race on the shared GPFS mount.
with socket.socket() as port_probe:
    port_probe.bind(('127.0.0.1', 0))
    os.environ['RAY_GCS_SERVER_PORT'] = str(port_probe.getsockname()[1])
os.environ['WANDB_DISABLE_CODE'] = 'true'
os.environ['WANDB_DISABLE_GIT'] = 'true'
# A newly created local Ray cluster inherits the private process environment.
# Do not serialize the key into runtime_env: upstream prints ray.init kwargs.
import ray
assert ray.__version__ == '2.50.0', ('Unexpected Ray version', ray.__version__)
print('ATI runtime Ray', ray.__version__, flush=True)
from verl.trainer import constants_ppo
constants_ppo.WANDB_RAY_ENV_PASSTHROUGH[:] = [
    name for name in constants_ppo.WANDB_RAY_ENV_PASSTHROUGH if name != 'WANDB_API_KEY'
]
assert 'WANDB_API_KEY' not in constants_ppo.get_ppo_ray_runtime_env()['env_vars']
# Pin each local service to a distinct available port. Ray 2.54 otherwise
# discovers agent ports through files with a short GPFS-sensitive timeout.
from ray._private.parameter import RayParams
_original_ray_params_init = RayParams.__init__
def _ray_params_with_explicit_ports(self, *args, **kwargs):
    sockets = []
    selected = {int(os.environ['RAY_GCS_SERVER_PORT'])}
    try:
        for name in ['metrics_agent_port', 'metrics_export_port', 'dashboard_agent_listen_port',
                     'runtime_env_agent_port', 'node_manager_port', 'object_manager_port']:
            if kwargs.get(name) in (None, 0):
                while True:
                    probe = socket.socket()
                    probe.bind(('0.0.0.0', 0))
                    port = probe.getsockname()[1]
                    if port not in selected:
                        break
                    probe.close()
                sockets.append(probe)
                selected.add(port)
                kwargs[name] = port
        return _original_ray_params_init(self, *args, **kwargs)
    finally:
        for probe in sockets:
            probe.close()
RayParams.__init__ = _ray_params_with_explicit_ports
from verl.trainer import main_ppo
sys.argv = [sys.argv[0], *overrides]
main_ppo.main()
