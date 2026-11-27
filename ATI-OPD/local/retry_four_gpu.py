"""Prepare a fresh attempt without overwriting earlier run artifacts."""
from pathlib import Path
import argparse,datetime,json,subprocess,shlex
ROOT=Path('/workspace/fudan/donghaosen/ATI-OPD')
p=argparse.ArgumentParser();p.add_argument('--reason',required=True);p.add_argument('--updates',default='{}');args=p.parse_args()
updates=json.loads(args.updates)
active=ROOT/'local/active-four-gpu-run.json';oldpath=Path(json.loads(active.read_text())['manifest']);old=json.loads(oldpath.read_text())
assert old['status'] in ['failed_queue_paused','stopped_memory_limit','stopped_gpu_busy'],old['status']
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');name='paper4gpu_'+stamp
code=ROOT/'local/runs'/name;code.mkdir(parents=True)
data=Path('/data/fd/donghaosen/ati-opd/runs')/name;data.mkdir(parents=True)
m={k:v for k,v in old.items() if k not in ['runs','updated_at_utc']}
m.update(batch=name,data_root=str(data),status='prepared',previous_attempt=str(oldpath),retry_reason=args.reason,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
m['runs']=[]
for i,prior in enumerate(old['runs']):
 variant=prior['variant'];out=data/variant
 for folder in ['logs','checkpoints','rollouts','tests','tmp','swanlab']:(out/folder).mkdir(parents=True)
 short=Path('/data/fd/donghaosen/ati-opd/tmp')/(stamp[-7:-1]+str(i));short.mkdir()
 cfg=json.loads(Path(prior['config']).read_text());cfg.update(updates)
 cfg.update({'trainer.logger':['console','swanlab'],'trainer.experiment_name':variant+'-'+stamp,'trainer.default_local_dir':str(out/'checkpoints'),'trainer.test_data_dir':str(out/'tests'),'trainer.metric_logging.archive.local_dir':str(out/'metric_archive'),'+ray_kwargs.ray_init._temp_dir':str(short/'ray')})
 config=code/(variant+'.json');config.write_text(json.dumps(cfg,indent=2)+'\n')
 m['runs'].append({'variant':variant,'tag':variant+'-'+stamp,'config':str(config),'output_dir':str(out),'session':f'ati-{variant}-{stamp}','short_tmp_dir':str(short),'logging_backend':'swanlab','state':'prepared'})
path=code/'manifest.json';path.write_text(json.dumps(m,indent=2)+'\n');active.write_text(json.dumps({'manifest':str(path)},indent=2)+'\n')
session='ati-campaign-'+stamp
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),'bash --noprofile --norc'],check=True)
command='source local/activate.sh && '+shlex.join([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'local/run_four_gpu_campaign.py'),'--manifest',str(path)])+' > '+shlex.quote(str(data/'campaign.log'))+' 2>&1'
subprocess.run(['tmux','send-keys','-t',session,'-l',command],check=True);subprocess.run(['tmux','send-keys','-t',session,'Enter'],check=True)
print(path)
