"""Evaluate upstream's configuration expressions without its job-launch operations."""
import json
import os
from pathlib import Path
import subprocess

from hydra.core.override_parser.overrides_parser import OverridesParser

ROOT = Path('/workspace/fudan/donghaosen/ATI-OPD')
common = (ROOT/'run/common.sh').read_text()
settings_start = common.index('    TRAIN_DATASET=${TRAIN_DATASET:-dapo_math_17k}')
settings_end = common.index('\n    export WANDB_MODE=', settings_start)
arguments_start = common.index('    local -a hydra_args=(')
arguments_end = common.index('\n    local runtime_env_json', arguments_start)
for variant, filename in [
    ('ati', 'dmpd--train_on_teacher--loss_opd--prune_future.sh'),
    ('vanilla', 'vanilla-opd.sh'),
]:
    method = (ROOT/'run'/filename).read_text()
    method = method[method.index('DEFAULT_EXPERIMENT_NAME='):method.index('# shellcheck source=run/common.sh')]
    script = '\n'.join([
        'set -e',
        f'source {ROOT}/run/common.sh',
        f'WORKSPACE={ROOT}',
        'DATASET_DIR=/data/fd/donghaosen/ati-opd',
        'VC_WORKER_NUM=2',
        method,
        'capture_configuration() {',
        common[settings_start:settings_end],
        common[arguments_start:arguments_end],
        'printf "%s\\0" "${hydra_args[@]}"',
        '}',
        'capture_configuration',
    ])
    # Do not inherit model, dataset, W&B, or experiment overrides from the caller.
    env = {'PATH': os.environ['PATH'], 'LANG': 'C.UTF-8'}
    result = subprocess.run(['bash', '--noprofile', '--norc'], input=script.encode(),
                            capture_output=True, env=env, cwd=ROOT, check=True)
    args = [value.decode() for value in result.stdout.split(b'\0') if value]
    parsed = OverridesParser.create().parse_overrides(args)
    values = {('++' if item.is_force_add() else '+' if item.is_add() else '') + item.key_or_group:
              item.value() for item in parsed}
    (ROOT/f'local/upstream-{variant}-overrides.json').write_text(json.dumps(values, indent=2)+'\n')
    (ROOT/f'local/upstream-{variant}-configuration-only.sh').write_text(script)
    print(variant, len(values), 'upstream overrides captured; no job launcher executed')
