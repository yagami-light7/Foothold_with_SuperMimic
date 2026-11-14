"""Import an isolated Roban simulation asset, preserving source physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

os.environ['SUPERMIMIC_ROBOT']='roban'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from supermimic.roban import SOURCE_MJCF,USD_DIR,USD_PATH,checkpoint_contract,model_contract


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--force',action='store_true');a=p.parse_args()
    expected=dict(**checkpoint_contract(),sim_mjcf_sha256=hashlib.sha256(SOURCE_MJCF.read_bytes()).hexdigest())
    manifest=USD_DIR/'contract.json'
    if USD_PATH.exists() and manifest.exists() and not a.force:
        saved=json.loads(manifest.read_text())
        if any(saved.get(k)!=v for k,v in expected.items()): raise ValueError('Existing Roban asset contract mismatch; use --force to rebuild generated asset')
        print(USD_PATH);return
    from pxr import Plug,Usd,UsdPhysics
    import isaacsim.asset.importer.utils.impl as iu
    Plug.Registry().RegisterPlugins(str(Path(iu.__file__).parent/'schemas'))
    from isaacsim.asset.importer.mjcf import MJCFImporter,MJCFImporterConfig
    source,_=model_contract()
    with tempfile.TemporaryDirectory(prefix='roban_usd_') as tmp:
        tmp=Path(tmp);tree=ET.parse(SOURCE_MJCF);root=tree.getroot()
        root.find('compiler').set('meshdir',str((SOURCE_MJCF.parent/'../meshes').resolve()))
        for tag in ('actuator','sensor'):
            element=root.find(tag)
            if element is not None: root.remove(element)
        free=root.find('.//joint[@type="free"]');free.set('name','floating_base_joint')
        sim_xml=tmp/SOURCE_MJCF.name;tree.write(sim_xml,encoding='unicode')
        sim=mujoco.MjModel.from_xml_path(str(sim_xml))
        for field in ('body_pos','body_quat','body_parentid','body_mass','body_inertia','body_ipos','body_iquat',
                      'jnt_type','jnt_axis','jnt_pos','jnt_range','qpos0','dof_armature','geom_contype','geom_conaffinity'):
            np.testing.assert_array_equal(getattr(source,field),getattr(sim,field),err_msg=field)
        cfg=MJCFImporterConfig(mjcf_path=str(sim_xml),usd_path=str(tmp/'converted'))
        generated=Path(MJCFImporter(cfg).import_mjcf())
        stage=Usd.Stage.Open(str(generated))
        for prim in list(stage.Traverse()):
            variant=prim.GetVariantSets().GetVariantSet('Physics')
            if 'mujoco' in variant.GetVariantNames(): variant.SetVariantSelection('mujoco')
        joints=[prim.GetName() for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
        from supermimic.roban import JOINT_NAMES
        if set(joints)!=set(JOINT_NAMES): raise ValueError(f'Imported USD joint mismatch: {joints}')
        if not any(prim.GetVariantSets().GetVariantSet('Physics').GetVariantSelection()=='mujoco' or
                   'mujoco' in prim.GetVariantSets().GetVariantSet('Physics').GetVariantNames() for prim in stage.Traverse()):
            raise ValueError('Roban USD lacks Newton/MuJoCo physics variant')
        USD_PATH.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(generated.parent,USD_PATH.parent,dirs_exist_ok=True)
    manifest.write_text(json.dumps(dict(**expected,source_motors_removed=source.nu,collision_masks='unchanged',
        self_collisions=True,ground='single scene plane',kinematics_and_inertias='exact'),indent=2)+'\n')
    print(USD_PATH)


if __name__=='__main__':main()
