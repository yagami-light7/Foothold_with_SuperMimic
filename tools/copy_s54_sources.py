"""Copy only S54 assets/reference into this repository; never overwrite different files."""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from supermimic.s54 import REPO


def copy_same_or_new(src,dst):
    if not src.is_file():raise FileNotFoundError(src)
    if dst.exists():
        if hashlib.sha256(src.read_bytes()).digest()!=hashlib.sha256(dst.read_bytes()).digest():
            raise ValueError(f'Refusing overwrite of different local file: {dst}')
    else:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gmr',type=Path,default=REPO.parent/'Leju-GMR');a=p.parse_args()
    src=a.gmr/'assets/kuavo_s54';dst=REPO/'data/s54_description'
    copy_same_or_new(src/'xml/biped_s54.xml',dst/'xml/biped_s54.xml')
    copy_same_or_new(src/'urdf/biped_s54.urdf',dst/'urdf/biped_s54.urdf')
    for mesh in (src/'meshes').rglob('*'):
        if mesh.is_file():copy_same_or_new(mesh,dst/'meshes'/mesh.relative_to(src/'meshes'))
    motions=sorted((a.gmr/'assets/lafan1_bvh').glob('*.bvh'))
    if not motions:raise FileNotFoundError(a.gmr/'assets/lafan1_bvh')
    for motion in motions:copy_same_or_new(motion,REPO/'data/retarget_data/raw'/motion.name)
    print(f'PASS source assets and {len(motions)} raw BVHs copied/verified; no external writes')


if __name__=='__main__':main()
