"""Synchronize one verified ATI experiment's terminal cloud state."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--state', choices=['ABORTED', 'CRASHED', 'FINISHED'], required=True)
args = parser.parse_args()
output = Path(args.output)
log = output / 'logs/train.log'
urls = re.findall(r'https://swanlab\.(?:cn|ai)/@([\w.-]+)/([\w.+-]+)/runs/([\w-]+)', log.read_text(errors='replace'))
if not urls:
    raise SystemExit(0)
cfg = json.loads(Path(args.config).read_text())
owner, project, slug = urls[-1]
assert owner == 'Dhs2004' and project == cfg['trainer.project_name']
os.environ['SWANLAB_ROOT'] = '/workspace/fudan/donghaosen/.credentials/swanlab-ati-opd'
os.environ['SWANLAB_CONFIG_DIR'] = os.environ['SWANLAB_ROOT']
import swanlab
key = Path('/workspace/fudan/donghaosen/.credentials/swanlab-20261031-4b.key').read_text().strip()
api = swanlab.Api(api_key=key)
path = f'{owner}/{project}/{slug}'
run = api.run(path)
assert run.name == cfg['trainer.experiment_name'], 'Cloud identity mismatch'
before = run.state
now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
if before != args.state:
    response = run._ctx.client.put(f'/project/{owner}/{project}/runs/{slug}/state',
        data={'state': args.state, 'finishedAt': now, 'from': 'sdk'})
    assert response.raw.ok
after = swanlab.Api(api_key=key).run(path).state
assert after == args.state
(output / 'logs/cloud-final-state.json').write_text(json.dumps(
    {'path': path, 'previous_state': before, 'verified_state': after, 'verified_at': now}, indent=2) + '\n')
print(f'SwanLab {path}: {after}', flush=True)
