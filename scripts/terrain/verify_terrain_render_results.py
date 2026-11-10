#!/usr/bin/env python3
"""Verify saved real Newton A/B evidence; never turn failed height tests into PASS.

Run with Isaac Lab Python for pxr/trimesh, no GPU simulation needed here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import trimesh

from terrain_render_diagnostics import compare_surfaces, load_mesh, sha256


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    summary = dict(assets=[], drop_regressions=[], normal_evidence=[])
    for asset_name, case_name, locations in [
        ('stairs_straight_4_v1', 'C_straight_final_matte', [-2., -1.3, -.9, -.5, 0.]),
        ('stairs_pyramid_4_v1', 'C_pyramid_final_matte', [0.]),
    ]:
        asset = Path('assets/terrains')/asset_name
        initial = read(root/f'{asset_name}_geometry.json')
        preserved = {name: sha256(asset/name) == digest for name, digest in initial['files'].items()}
        assert all(preserved.values()), f'An original asset changed: {asset_name}'
        mesh = load_mesh(asset)
        stage = Usd.Stage.Open(str(root/case_name/'scene.usda'))
        collider = UsdGeom.Mesh.Get(stage, '/World/Target/mesh')
        visual = UsdGeom.Mesh.Get(stage, '/World/Target/flat_visual')
        points = np.asarray(collider.GetPointsAttr().Get())
        faces = np.asarray(collider.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        visual_points = np.asarray(visual.GetPointsAttr().Get())
        visual_faces = np.asarray(visual.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        assert collider.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        assert not visual.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        assert np.array_equal(points, np.asarray(mesh.vertices, dtype=np.float32))
        assert np.array_equal(faces, mesh.faces)
        assert np.array_equal(visual_points[visual_faces], points[faces])
        assert collider.ComputeVisibility() == UsdGeom.Tokens.invisible
        assert visual.ComputeVisibility() != UsdGeom.Tokens.invisible
        # The visual copy has the exact same triangle corners, not just a similar
        # top height. Also check all exported height-map centers after USD float32.
        visual_mesh = trimesh.Trimesh(visual_points, visual_faces, process=False)
        surface = compare_surfaces(mesh, visual_mesh, asset)
        assert surface['same_coverage']
        assert surface['max_top_height_difference_m'] < 1e-6
        summary['assets'].append(dict(asset=asset_name, original_files_unchanged=preserved,
            collider_points_exact_float32=True, collider_indices_identical=True,
            visual_triangles_identical_to_collider=True, surface_comparison=surface))
        for index, x in enumerate(locations):
            reports, traces, shapes = [], [], []
            for suffix in ['before', 'after']:
                directory = root/f'drop_{asset_name}_{index}_{suffix}'
                report = read(directory/'diagnosis.json')
                assert report.get('drop', {}).get('steps') == 600, directory
                reports.append(report)
                traces.append(np.load(directory/'drop_trajectory.npz')['position_w'])
                shapes.append(next(s for s in report['model'] if s['path'] == '/World/Target/mesh'))
            vertices_same = shapes[0]['vertices_sha256'] == shapes[1]['vertices_sha256']
            indices_same = shapes[0]['indices_sha256'] == shapes[1]['indices_sha256']
            delta = float(np.max(np.abs(traces[0]-traces[1])))
            assert vertices_same and indices_same and delta <= 1e-6
            # Preserve the two independent verdicts: no physics regression does
            # NOT imply that the original low-step contact geometry is correct.
            summary['drop_regressions'].append(dict(asset=asset_name, drop_x=x,
                collider_vertices_identical=vertices_same, collider_indices_identical=indices_same,
                max_600_step_position_difference_m=delta, unchanged_physics=True,
                before=reports[0]['drop'], after=reports[1]['drop']))
    for name in ['A_native_box', 'B_imported_box_v3', 'B_flat_corners',
                 'C_straight_original', 'D_straight_exterior', 'D_straight_flat',
                 'C_pyramid_original', 'D_pyramid_original_normals', 'D_pyramid_flat',
                 'C_straight_flat_v2_visuals', 'C_pyramid_flat_v2_visuals',
                 'C_straight_final_matte', 'C_pyramid_final_matte']:
        report = read(root/name/'diagnosis.json')
        assert report['status'] == 'PASS', name
        # GL mesh IDs match the shape indices used by Newton Viewer.log_geo().
        visible = next(i for i, s in enumerate(report['model']) if s['visible'] and s['path'].startswith('/World/Target'))
        geometry = next(g for g in report['gl_meshes'] if g['name'].endswith(f'_{visible}'))
        if name.startswith('A_') or 'flat' in name or 'matte' in name:
            assert geometry['corner_normal_mismatch_count'] == 0, name
        summary['normal_evidence'].append(dict(case=name, **geometry))
    summary['usd_dome_0_vs_800_images_identical'] = (
        (root/'C_straight_original/newton_gl.png').read_bytes() ==
        (root/'C_straight_no_dome/newton_gl.png').read_bytes())
    assert summary['usd_dome_0_vs_800_images_identical']
    summary['render_fix_and_unchanged_physics'] = 'PASS'
    summary['default_contact_all_treads_correct'] = all(r['after']['passed'] for r in summary['drop_regressions'])
    summary['diagnostic_newton_contacts'] = read(root/'diagnostic_newton_contacts_low_step/diagnosis.json')['drop']
    with (root/'verification.json').open('w') as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
