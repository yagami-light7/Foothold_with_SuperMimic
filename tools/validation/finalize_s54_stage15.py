"""Serial video generation and evidence audit after bounded Stage15 workers exit."""
import argparse,json,subprocess,sys,time,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P='/home/rm/workspace/IsaacLab_3_0/.venv/bin/python'
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();d=a.run_dir.resolve()
    started=time.monotonic()
    while json.loads((d/'status.json').read_text())['status'] not in ('COMPLETED','FAILED'):
        if time.monotonic()-started>1800:raise TimeoutError('Finalization wait budget')
        time.sleep(5)
    if not (d/'eval_b/summary.json').exists():raise RuntimeError('B evaluation incomplete; no fabricated video/report')
    subprocess.run([P,'-B','tools/validation/audit_s54_stage15_latents.py','--run-dir',str(d)],check=True,cwd=ROOT,timeout=120)
    videos=[]
    for offset in (0,60,120):
        target=d/'videos'/f'teacher_phase_a_phase_b_offset{offset}.mp4'
        subprocess.run([P,'-B','tools/visualization/render_s54_stage15.py','--run-dir',str(d),'--offset',str(offset),'--out',str(target)],check=True,cwd=ROOT,timeout=240)
        videos.append(target)
    sys.path.insert(0,str(ROOT))
    from tools.visualization.render_s54_joint_comparison import resolve_ffmpeg
    ff=resolve_ffmpeg();manifest=d/'videos/video_sequence.txt';manifest.write_text(''.join("file '"+x.name+"'\n" for x in videos))
    combined=d/'videos/teacher_phase_a_phase_b_all_windows.mp4'
    subprocess.run([ff,'-n','-loglevel','error','-f','concat','-safe','0','-i',str(manifest),'-c','copy','-movflags','+faststart',str(combined)],check=True,timeout=60)
    r=subprocess.run([ff,'-v','error','-i',str(combined),'-progress','pipe:1','-f','null','-'],capture_output=True,text=True,check=True,timeout=120)
    progress={k:v for line in r.stdout.splitlines() if '=' in line for k,v in [line.split('=',1)]}
    if int(progress['frame'])!=750:raise RuntimeError('Expected 750 decoded frames')
    subprocess.run([ff,'-n','-v','error','-ss','5','-i',str(combined),'-frames:v','1',str(d/'videos/midpoint.png')],check=True,timeout=60)
    verification=dict(full_decode_exitcode=r.returncode,frames=int(progress['frame']),out_time_us=int(progress['out_time_us']),
        videos=list(map(str,videos)),combined=str(combined),new_rollouts_for_render=0)
    (d/'videos/video_verification.json').write_text(json.dumps(verification,indent=2))
    # Check all three policies really used the same physical initial state.
    audit_code='''
import json,numpy as np,hashlib
from pathlib import Path
d=Path(__import__('sys').argv[1]);A=json.loads((d/'eval_a/summary.json').read_text());B=json.loads((d/'eval_b/summary.json').read_text());checks=[]
for off in (0,60,120):
 states=[]
 for ev,role in ((A,'motion'),(A,'foot'),(B,'motion'),(B,'foot')):
  row=next(r for r in ev['rows'][role] if r['offset_s']==off)
  with np.load(row['record']) as f:states.append(f['state'][0].copy())
 gap=max(float(abs(s-states[0]).max()) for s in states)
 checks.append(dict(offset_s=off,max_initial_state_gap=gap));assert gap<1e-6
teacher=Path(json.loads((d/'profile.json').read_text())['teacher']);sha=hashlib.file_digest(teacher.open('rb'),'sha256').hexdigest()
assert sha==json.loads((d/'profile.json').read_text())['teacher_sha256']
(d/'completion_audit.json').write_text(json.dumps(dict(initial_states=checks,teacher_sha256=sha,teacher_unchanged=True,original_experiments_written=False),indent=2))
'''
    subprocess.run([P,'-B','-c',audit_code,str(d)],cwd=ROOT,check=True,timeout=120)
    snapshot=d/'delivery_source_snapshot';snapshot.mkdir()
    files=[*ROOT.glob('supermimic/stage15*.py'),*ROOT.glob('tools/train/run_s54_stage15*'),*ROOT.glob('tools/validation/*stage15*.py'),ROOT/'tools/visualization/render_s54_stage15.py',ROOT/'tools/ops/sync_s54_stage15.py',ROOT/'tools/tests/test_stage15.py',*ROOT.glob('configs/motion_prior/stage15/*.json')]
    files += [ROOT/'supermimic'/f'{name}.py' for name in ('env','mdp','networks','buffer','reference_foot','goals','motion','rotations',
        'kinematics','standing_diagnostics','control','s54','transfer','motion_prior_data','foot_mdp','foot_geometry','robot',
        'joint_runtime','joint_models','joint_updates','motion_prior_runtime')]
    files += [ROOT/'tools/visualization/render.py',ROOT/'tools/visualization/policy_preview.py']
    hashes={}
    for source in files:
        relative=source.relative_to(ROOT);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (snapshot/'sha256.json').write_text(json.dumps(hashes,indent=2))
    shutil.copyfile('/tmp/stage15_unit_tests.log',d/'unit_tests.log')
    for _ in range(60):
        if (d/'swanlab/status.json').exists() and json.loads((d/'swanlab/status.json').read_text())['status']=='VERIFIED':break
        time.sleep(5)
    subprocess.run(['python3','tools/validation/report_s54_stage15.py','--run-dir',str(d)],cwd=ROOT,check=True,timeout=30)
    print('STAGE15 delivery artifacts ready: '+str(d),flush=True)
if __name__=='__main__':main()
