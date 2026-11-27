"""Sequential paper runs on GPUs 0-3, with tmux job control and measured VRAM guard."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

ROOT = Path('/workspace/fudan/donghaosen/ATI-OPD')
parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--resume-monitor', action='store_true')
args = parser.parse_args()
path = Path(args.manifest)
m = json.loads(path.read_text())
if not os.environ.get('TMUX') or not m['ports_authorized_by_user']:
    raise SystemExit('tmux and explicit port authorization are required.')
import pynvml
pynvml.nvmlInit()
handles = [pynvml.nvmlDeviceGetHandleByUUID(uuid) for uuid in m['gpu_uuids']]
# vLLM 0.18 requires numeric device IDs. Verify their stable UUIDs before use.
assert [pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(4)] == m['gpu_uuids']
bus_ids = [pynvml.nvmlDeviceGetPciInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).busId for i in range(pynvml.nvmlDeviceGetCount())]
assert bus_ids == sorted(bus_ids), 'PCI_BUS_ID ordering differs from NVML indices'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save():
    m['updated_at_utc'] = now()
    path.write_text(json.dumps(m, indent=2) + '\n')

def finalize_cloud(run, state):
    # Use the existing API-capable SDK; training keeps its compatible SDK pinned.
    python = '/workspace/fudan/donghaosen/Self-Distillation-AgentRL/local_assets/conda/envs/agentrl-reskill/bin/python'
    try:
        result = subprocess.run([python, str(ROOT/'local/finalize_swanlab.py'),
            '--config', run['config'], '--output', run['output_dir'], '--state', state],
            timeout=60, capture_output=True, text=True)
        run['cloud_finalization'] = 'verified_or_no_cloud_run' if result.returncode == 0 else 'needs_retry'
        print(result.stdout, flush=True)
        if result.returncode:
            print('SwanLab terminal-state synchronization needs retry.', flush=True)
    except subprocess.TimeoutExpired:
        run['cloud_finalization'] = 'needs_retry'

def owned_processes(tag):
    result = []
    target = ('ATI_RUN_TAG=' + tag).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if not os.readlink(proc/'cwd').startswith(str(ROOT)):
                continue
            raw = (proc/'environ').read_bytes().split(b'\0')
            if target not in raw:
                continue
            stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0] != 'Z':
                result.append((int(proc.name), stat[19]))
        except (OSError, ProcessLookupError):
            pass
    return result

def stop_own_run(run, reason):
    # Stop the verified foreground training job using the shared-machine rule.
    if not (Path(run['output_dir'])/'logs/train.exit').exists():
        subprocess.run(['tmux','send-keys','-t',run['session'],'C-z'],check=True)
        time.sleep(1)
        subprocess.run(['tmux','send-keys','-t',run['session'],'-l','kill %1'],check=True)
        subprocess.run(['tmux','send-keys','-t',run['session'],'Enter'],check=True)
        time.sleep(2)
    # Ray services may have detached from the shell job; stop only tagged descendants.
    for pid, start in owned_processes(run['tag']):
        try:
            stat = Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
            if stat[19] == start:
                os.kill(pid, signal.SIGTERM)
                if stat[0] in ('T','t'):
                    os.kill(pid, signal.SIGCONT)
        except (OSError, ProcessLookupError):
            pass
    run['state'] = reason

if m['status'] != 'prepared' and not (args.resume_monitor and m['status'] == 'running'):
    raise SystemExit('Refusing to launch an already-used campaign manifest.')
m['status'] = 'running'
save()
for run in m['runs']:
    if run['state'] == 'completed':
        continue
    resuming = args.resume_monitor and run['state'] in ('initializing', 'training')
    if resuming:
        assert subprocess.run(['tmux','has-session','-t',run['session']],capture_output=True).returncode == 0
    if not resuming and any(pynvml.nvmlDeviceGetComputeRunningProcesses(h) for h in handles):
        run['state'] = 'blocked_gpu_busy'
        m['status'] = 'stopped_gpu_busy'
        save()
        raise SystemExit('Assigned GPUs are not free; no unrelated process is stopped.')
    output = Path(run['output_dir'])
    log = output/'logs/train.log'
    exit_file = output/'logs/train.exit'
    if not resuming:
        if exit_file.exists() or subprocess.run(['tmux','has-session','-t',run['session']],capture_output=True).returncode == 0:
            raise SystemExit('Refusing to reuse an existing experiment session/output.')
        subprocess.run(['tmux','new-session','-d','-s',run['session'],'-c',str(ROOT),'bash --noprofile --norc'],check=True)
        env = {
            'CUDA_VISIBLE_DEVICES': '0,1,2,3',
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'ATI_RUN_TAG':run['tag'], 'ATI_INTERNAL_PORTS_AUTHORIZED':'1',
            'PYTHONPATH':str(ROOT/'local/runtime-ray250') + os.pathsep + str(ROOT), 'PYTHONNOUSERSITE':'1', 'PYTHONDONTWRITEBYTECODE':'1',
            'WANDB_MODE':'disabled','SWANLAB_MODE':'cloud',
            'SWANLAB_LOG_DIR':str(output/'swanlab'),
            'SWANLAB_ROOT':'/workspace/fudan/donghaosen/.credentials/swanlab-ati-opd',
            'SWANLAB_SAVE_DIR':'/workspace/fudan/donghaosen/.credentials/swanlab-ati-opd',
            'SWANLAB_CONFIG_DIR':'/workspace/fudan/donghaosen/.credentials/swanlab-ati-opd',
            'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','VLLM_HOST_IP':'127.0.0.1','MASTER_ADDR':'127.0.0.1',
            'NCCL_SOCKET_IFNAME':'lo','GLOO_SOCKET_IFNAME':'lo','NCCL_IB_DISABLE':'1',
            'RAY_USAGE_STATS_ENABLED':'0','RAY_DEDUP_LOGS':'0',
            'TMPDIR':run['short_tmp_dir'],
        }
        command = 'source local/activate.sh && unset RAY_ADDRESS WANDB_RESUME WANDB_API_KEY; export ' + ' '.join(
            name+'='+shlex.quote(value) for name,value in env.items())
        command += '; ' + shlex.join([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'local/run_four_gpu_entry.py'),'--config',run['config']])
        command += ' > '+shlex.quote(str(log))+' 2>&1; printf "%s\\n" "$?" > '+shlex.quote(str(exit_file))
        subprocess.run(['tmux','send-keys','-t',run['session'],'-l',command],check=True)
        subprocess.run(['tmux','send-keys','-t',run['session'],'Enter'],check=True)
        run.update(state='initializing',started_at_utc=now(),console_log=str(log),peak_mib=[0]*4)
        save()
    last_save = 0
    with (output/'logs/gpu-memory.jsonl').open('a',buffering=1) as memory_log:
        while True:
            # Apply explicit user limit changes without restarting training.
            limits_path = ROOT/'local/four-gpu-memory-limits.json'
            if limits_path.exists():
                limits = json.loads(limits_path.read_text())
                m['memory_stop_mib_per_gpu'] = limits['memory_stop_mib_per_gpu']
            used=[round(pynvml.nvmlDeviceGetMemoryInfo(h).used/2**20,1) for h in handles]
            run['peak_mib']=[max(old,new) for old,new in zip(run['peak_mib'],used)]
            memory_log.write(json.dumps({'utc':now(),'used_mib':used,'peak_mib':run['peak_mib']})+'\n')
            if max(used)>m['memory_stop_mib_per_gpu']:
                stop_own_run(run,'stopped_memory_limit')
                finalize_cloud(run,'ABORTED')
                m['status']='stopped_memory_limit'
                save()
                raise SystemExit('VRAM guard stopped only this ATI run; remaining queue paused.')
            if log.exists():
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-131072)); tail=f.read().decode(errors='replace')
                urls=re.findall(r'https://swanlab\.(?:cn|ai)/@[\w.-]+/[\w.+-]+/runs/[\w-]+',tail)
                if urls:run['swanlab_url']=urls[-1]
                steps=re.findall(r'\bstep:(\d+)\s*-',tail)
                if steps:
                    run['last_logged_step']=int(steps[-1]);run['state']='training'
            if time.monotonic()-last_save>10:
                save();last_save=time.monotonic()
            if exit_file.exists():
                try:code=int(exit_file.read_text().strip())
                except ValueError:
                    time.sleep(1);continue
                run['exit_code']=code;run['finished_at_utc']=now()
                if code:
                    stop_own_run(run,'failed')
                    finalize_cloud(run,'ABORTED' if code in (143,148) else 'CRASHED')
                    m['status']='failed_queue_paused'
                    save()
                    raise SystemExit(f"{run['variant']} failed with exit code {code}; inspect {log}")
                finalize_cloud(run,'FINISHED')
                run['state']='completed';save()
                break
            time.sleep(0.5)
    # Let the completed local Ray cluster release GPUs before the next run.
    for _ in range(30):
        if not any(pynvml.nvmlDeviceGetComputeRunningProcesses(h) for h in handles):break
        time.sleep(2)
m['status']='completed'
save()
