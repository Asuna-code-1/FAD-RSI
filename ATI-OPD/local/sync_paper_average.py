"""Backfill and follow Table 1 Average without resuming or closing a live run."""
import argparse
import datetime
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path('/workspace/fudan/donghaosen/ATI-OPD')
os.environ['SWANLAB_ROOT'] = '/workspace/fudan/donghaosen/.credentials/swanlab-ati-opd'
os.environ['SWANLAB_CONFIG_DIR'] = os.environ['SWANLAB_ROOT']
spec = importlib.util.spec_from_file_location('paper_average', ROOT/'verl/utils/evaluation/paper_math_average.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
KEY = helper.PAPER_AVERAGE_KEY
SOURCES = helper.PAPER_MATH_KEYS


def scalar_series(response):
    return {
        series['key']: {int(p['index']): p for p in series.get('metrics', [])}
        for series in response.get('list', [])
    }


def synchronize(api, entry):
    url = entry.get('swanlab_url')
    if not url:
        return True
    path = url.split('/@', 1)[1].replace('/runs/', '/')
    assert path.startswith('Dhs2004/ati-opd/')
    run = api.run(path)
    cfg = json.loads(Path(entry['config']).read_text())
    assert run.name == cfg['trainer.experiment_name'], 'Cloud identity mismatch'
    state_before = run.state
    response = run.metrics(keys=[*SOURCES, KEY], sample=1500)
    series = scalar_series(response)
    common = set.intersection(*(set(series.get(key, {})) for key in SOURCES))
    expected = {}
    pending = []
    for step in sorted(common):
        average = helper.paper_math_average({key: series[key][step]['data'] for key in SOURCES})
        if average is None:
            continue
        expected[step] = average
        existing = series.get(KEY, {}).get(step)
        if existing:
            assert math.isclose(float(existing['data']), average, abs_tol=1e-8), 'Existing Average differs'
            continue
        timestamp = max(series[key][step]['timestamp'] for key in SOURCES)
        created = datetime.datetime.fromtimestamp(timestamp/1000, datetime.timezone.utc).isoformat()
        pending.append({'key': KEY, 'index': step, 'epoch': step,
                        'data': average, 'create_time': created})
    if pending:
        client = run._ctx.client
        result = client.post(f'/experiment/{run.run_id}/columns', data=[{
            'class': 'CUSTOM', 'type': 'FLOAT', 'key': KEY, 'name': 'Average',
            'sectionName': 'test-legacy-core', 'sectionType': 'PUBLIC',
            'chartName': 'Average (%)',
        }])
        assert result.raw.ok, f'Column registration failed: HTTP {result.raw.status_code}'
        result = client.post('/house/metrics', data={
            'projectId': run.project_id, 'experimentId': run.run_id,
            'type': 'scalar', 'flagId': None, 'metrics': pending,
        })
        assert result.raw.ok, f'Scalar upload failed: HTTP {result.raw.status_code}'
        print(json.dumps({'run': path, 'submitted': {p['index']: p['data'] for p in pending}}), flush=True)
        for attempt in range(6):
            actual = scalar_series(api.run(path).metrics(keys=[KEY], sample=1500)).get(KEY, {})
            if all(step in actual and math.isclose(float(actual[step]['data']), value, abs_tol=1e-8)
                   for step, value in expected.items()):
                break
            time.sleep(5)
        else:
            raise RuntimeError('Cloud readback has not yet confirmed all Average points')
    else:
        actual = series.get(KEY, {})
    report = {
        'metric': KEY, 'unit': 'percent', 'formula': '100 * sum(five legacy acc/mean@16) / 5',
        'source_keys': SOURCES, 'verified_points': expected,
        'cloud_state_before': state_before, 'cloud_state_after': api.run(path).state,
        'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    out = Path(entry['output_dir'])/'logs/paper-average-sync.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    if pending:
        print(f'Verified {len(expected)} Average points; run state {report["cloud_state_after"]}', flush=True)
    # A terminal run may still be flushing its last evaluation to the cloud.
    # Wait until every complete locally logged evaluation has appeared remotely.
    log = Path(entry['output_dir'])/'logs/train.log'
    local_steps = set()
    if log.exists():
        with log.open(errors='replace') as stream:
            for line in stream:
                match = re.search(r'\bstep:(\d+)\s*-', line)
                if match and all(key + ':' in line for key in SOURCES):
                    local_steps.add(int(match.group(1)))
    return local_steps.issubset(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    campaign = json.loads(manifest_path.read_text())
    lock = (Path(campaign['data_root'])/'paper-average-sync.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import swanlab
    api = swanlab.Api(api_key=Path('/workspace/fudan/donghaosen/.credentials/swanlab-20261031-4b.key').read_text().strip())
    done = set()
    terminal = {'completed', 'failed', 'stopped_memory_limit', 'stopped_by_user'}
    while True:
        campaign = json.loads(manifest_path.read_text())
        failed = False
        for entry in campaign['runs']:
            if entry['tag'] in done:
                continue
            try:
                caught_up = synchronize(api, entry)
                if entry['state'] in terminal and caught_up:
                    done.add(entry['tag'])
                elif entry['state'] in terminal:
                    failed = True
            except Exception as error:
                failed = True
                print(f'Average synchronization needs retry: {type(error).__name__}', flush=True)
                if args.once:
                    raise
        if args.once or (campaign['status'] != 'running' and not failed):
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
