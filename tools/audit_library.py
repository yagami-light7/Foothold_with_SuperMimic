"""CPU-only library inventory, Roban model compilation and CSV schema audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_library import sha256, write_json


def audit(verify_hashes=False):
    import mujoco
    import numpy as np

    library = ROOT / 'library'
    registry = json.loads((library / 'registry.json').read_text())
    robot = registry['robots']['roban_s22_0901']
    model_path = library / robot['mjcf']
    model = mujoco.MjModel.from_xml_path(str(model_path))
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or model.body(int(model.jnt_bodyid[free[0]])).name != 'base_link':
        raise ValueError('Roban must have one floating torso base_link')
    joints = [model.joint(j).name for j in range(model.njnt) if j not in free]
    if len(joints) != 21 or model.nu != 21 or any(n.startswith('zhead') for n in joints):
        raise ValueError('Expected the supplied fixed-head, 21-active-joint Roban')
    urdf = ET.parse(library / robot['urdf'])
    urdf_joints = [j.get('name') for j in urdf.findall('joint') if j.get('type') in ('revolute', 'continuous', 'prismatic')]
    if set(urdf_joints) != set(joints):
        raise ValueError('Roban URDF/MJCF active joint sets differ')
    compiler = ET.parse(model_path).getroot().find('compiler')
    mesh_dir = model_path.parent / compiler.get('meshdir', '.')
    meshes = ET.parse(model_path).getroot().findall('asset/mesh')
    for mesh in meshes:
        if not (mesh_dir / mesh.get('file')).is_file():
            raise FileNotFoundError(mesh.get('file'))
    root_columns = ['Frame', *[f'root_translate{a}' for a in 'XYZ'], *[f'root_rotate{a}' for a in 'XYZW']]
    required = root_columns + joints
    dataset_reports = {}
    for name, config in robot['datasets'].items():
        folder = library / config['path']
        files = sorted(folder.rglob('*.csv'))
        if len(files) != config['expected_clips']:
            raise ValueError(f'Unexpected CSV count for {name}: {len(files)}')
        extra_counts = set()
        for path in files:
            with path.open(newline='', encoding='utf-8-sig') as stream:
                reader = csv.reader(stream)
                header = next(reader)
            if header[:len(required)] != required or len(set(header)) != len(header):
                raise ValueError(f'CSV root/joint header differs from Roban model: {path}')
            extra_counts.add(len(header) - len(required))
        dataset_reports[name] = dict(clips=len(files), bytes=sum(p.stat().st_size for p in files),
                                     declared_fps=config['fps'], fps_provenance=config['fps_provenance'],
                                     extra_diagnostic_columns=sorted(extra_counts), csv_schema='PASS')
    imported = json.loads((library / 'manifests/roban_import.json').read_text())
    for row in imported['entries']:
        path = library / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError(f'Missing/truncated imported asset: {path}')
        if verify_hashes and sha256(path) != row['sha256']:
            raise ValueError(f'Changed imported asset: {path}')
    migration = json.loads((library / 'manifests/data_migration.json').read_text())
    for row in migration['entries']:
        if sha256(ROOT / row['new']) != row['sha256']:
            raise ValueError(f'Changed migrated S54 asset: {row["new"]}')
    if (ROOT / 'data').exists():
        raise ValueError('Old data/ directory still exists')
    contract = dict(robot='roban_s22_0901', source_mjcf_sha256=sha256(model_path),
                    urdf_sha256=sha256(library / robot['urdf']), root_body='base_link',
                    root_semantics='floating chest/torso; waist and pelvis are descendants',
                    joint_names=joints, num_joints=len(joints), nq=model.nq, nv=model.nv, nu=model.nu,
                    body_names=[model.body(i).name for i in range(1, model.nbody)],
                    body_count_including_fixed_frames=model.nbody - 1,
                    positive_mass_body_count=int((model.body_mass[1:] > 0).sum()),
                    visual_mesh_references=len(meshes), model_mass_kg=float(model.body_mass.sum()),
                    mjcf_quaternion_order='wxyz', csv_root_quaternion_order='xyzw',
                    csv_root_columns=root_columns, csv_joint_columns=joints,
                    csv_extra_columns='retarget diagnostics, not robot actions')
    write_json(library / 'manifests/roban_model_contract.json', contract)
    report = dict(status='PASS', scope='asset import, model compilation, CSV headers and path migration only',
                  robot=contract, datasets=dataset_reports, imported_files=imported['files'],
                  imported_bytes=imported['bytes'], import_checksums='all compared with source during copy',
                  import_hashes_rechecked=verify_hashes, migrated_files_verified=len(migration['entries']),
                  old_data_directory_exists=False, newton_or_training_adaptation='not part of this migration',
                  timing_pending=[n for n, c in robot['datasets'].items() if c['fps'] is None])
    write_json(library / 'manifests/library_audit.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'scope', 'datasets', 'imported_files', 'migrated_files_verified', 'timing_pending']}, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-hashes', action='store_true', help='Also reread and hash the entire imported motion collection')
    audit(parser.parse_args().verify_hashes)
