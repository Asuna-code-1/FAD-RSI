"""Fork artifact paths and resume the last committed checkpoint; preserve prior evidence."""
import datetime
import json
from pathlib import Path
import shutil

ROOT = Path('/workspace/fudan/donghaosen/ATI-OPD')
old_path = Path(json.loads((ROOT/'local/active-four-gpu-run.json').read_text())['manifest'])
old = json.loads(old_path.read_text())
assert old['status'] == 'stopped_memory_limit'
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
name = 'paper4gpu_recovery_' + stamp
code = ROOT/'local/runs'/name
data = Path('/data/fd/donghaosen/ati-opd/runs')/name
code.mkdir(parents=True)
data.mkdir(parents=True)
manifest = {k: v for k, v in old.items() if k not in ('runs', 'updated_at_utc', 'notes')}
manifest.update(batch=name, data_root=str(data), status='prepared', previous_attempt=str(old_path),
                created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                retry_reason='Bounded FP32 selected-logprob forward/backward; resume committed step 120',
                memory_stop_mib_per_gpu=138240, runs=[], notes=[
                    'GPU 0-3: 2 student + 2 teacher; 135 GiB monitored stop threshold.',
                    'Preserve upstream algorithm, shuffle=False, batch64, n4, response23552, 217 total steps.',
                    'First run resumes model, optimizer, RNG, dataloader and ATI calibration from step120.',
                    'Step0 already measured in parent run; skip redundant pre-training evaluation on resume.',
                    'Original artifacts retained. New SwanLab run starts at step121; steps121-125 replayed.',
                ])
for index, prior in enumerate(old['runs']):
    cfg = json.loads(Path(prior['config']).read_text())
    variant = prior['variant']
    out = data/variant
    for folder in ('logs', 'checkpoints', 'tests', 'swanlab', 'rollouts'):
        (out/folder).mkdir(parents=True)
    short = Path('/data/fd/donghaosen')/('r'+stamp[9:15]+str(index))
    short.mkdir()
    cfg.update({'trainer.experiment_name': variant+'-aligned-'+('resume120-' if index==0 else 'step0-')+stamp,
                'trainer.default_local_dir':str(out/'checkpoints'), 'trainer.test_data_dir':str(out/'tests'),
                'trainer.metric_logging.archive.local_dir':str(out/'metric_archive'),
                '+ray_kwargs.ray_init._temp_dir':str(short/'ray')})
    if index == 0:
        parent_cfg = json.loads(Path(prior['config']).read_text())
        checkpoint = Path(parent_cfg['trainer.default_local_dir'])/'global_step_120'
        required = ['data.pt', 'metrics_commit.json', 'opd_prune_future_calibration.json']
        required += [f'actor/{kind}_world_size_2_rank_{rank}.pt' for rank in (0, 1)
                     for kind in ('model', 'optim', 'extra_state')]
        assert all((checkpoint/file).is_file() and (checkpoint/file).stat().st_size > 0 for file in required)
        cfg.update({'trainer.resume_mode':'resume_path', 'trainer.resume_from_path':str(checkpoint),
                    'trainer.test_before_train':False})
        # Copy the ledger to a new root. Prior committed shard paths remain readable,
        # preserving the commit hashes; new shards are written under the new root.
        shutil.copytree(parent_cfg['trainer.metric_logging.archive.local_dir'], out/'metric_archive')
        ledger_path = out/'metric_archive/manifest.json'
        ledger = json.loads(ledger_path.read_text())
        ledger['experiment'] = cfg['trainer.experiment_name']
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        (out/'logs/recovery-lineage.json').write_text(json.dumps({
            'parent_swanlab_url':prior['swanlab_url'], 'parent_checkpoint':str(checkpoint),
            'committed_step':120, 'last_logged_step':125, 'next_step':121,
            'initial_average_percent':27.761145104895103, 'step120_average_percent':38.8643648018648,
            'memory_fix':'FP32 selected-logprob workspace bounded to 512 tokens in forward and backward',
        },indent=2)+'\n')
    else:
        (out/'metric_archive').mkdir()
    cfg_path = code/(variant+'.json')
    cfg_path.write_text(json.dumps(cfg,indent=2)+'\n')
    tag = variant+'-recovery-'+stamp
    manifest['runs'].append({'variant':variant, 'tag':tag, 'config':str(cfg_path),
        'output_dir':str(out), 'short_tmp_dir':str(short), 'session':'ati-'+tag,
        'logging_backend':'swanlab', 'state':'prepared'})
path = code/'manifest.json'
path.write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'local/prepared-recovery-campaign.json').write_text(json.dumps({'manifest':str(path)},indent=2)+'\n')
print(path)
