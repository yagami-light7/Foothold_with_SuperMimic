"""CPU geometry diagnostics; never writes an exported terrain asset."""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_mesh(asset):
    with np.load(Path(asset)/'terrain_mesh.npz', allow_pickle=False) as data:
        return trimesh.Trimesh(data['vertices'], data['faces'], process=False)


def top_surface(mesh, xy):
    """Exact triangle intersections at XY; no interpolation across stair edges."""
    xy = np.asarray(xy)
    height = np.full(len(xy), -np.inf)
    for tri in mesh.triangles:
        a, b, c = tri
        e, f = b[:2]-a[:2], c[:2]-a[:2]
        den = e[0]*f[1]-e[1]*f[0]
        if abs(den) < 1e-14:
            continue
        d = xy-a[:2]
        u = (d[:, 0]*f[1]-d[:, 1]*f[0])/den
        v = (e[0]*d[:, 1]-e[1]*d[:, 0])/den
        hit = (u >= -1e-10) & (v >= -1e-10) & (u+v <= 1+1e-10)
        height = np.maximum(height, np.where(hit, a[2]+u*(b[2]-a[2])+v*(c[2]-a[2]), -np.inf))
    return height


def audit_mesh(mesh):
    from shapely.geometry import Polygon
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    used = np.unique(f)
    canonical = [tuple(sorted(map(tuple, np.round(t, 12)))) for t in mesh.triangles]
    dup = [n for n in Counter(canonical).values() if n > 1]
    components = mesh.split(only_watertight=False)
    welded = mesh.copy(); welded.merge_vertices()
    edge_counts = np.bincount(welded.edges_unique_inverse)
    hard_vertices = 0
    for index in used:
        normals = mesh.face_normals[np.any(f == index, axis=1)]
        if len(normals) > 1 and (normals @ normals.T).min() < .99:
            hard_vertices += 1
    overlap = []
    for i, j in combinations(range(len(f)), 2):
        ni, nj = mesh.face_normals[[i, j]]
        if abs(ni @ nj) < 1-1e-10:
            continue
        if np.max(np.abs((mesh.triangles[j]-mesh.triangles[i, 0]) @ ni)) > 1e-10:
            continue
        axis = int(np.argmax(np.abs(ni)))
        p = Polygon(np.delete(mesh.triangles[i], axis, axis=1))
        q = Polygon(np.delete(mesh.triangles[j], axis, axis=1))
        area = p.intersection(q).area
        if area > 1e-10:
            overlap.append(dict(faces=[i, j], area_m2=area, opposite=bool(ni @ nj < 0)))
    boxes = [x.bounds for x in components if x.is_watertight and len(x.vertices) == 8 and len(x.faces) == 12]
    penetrations = []
    for i, j in combinations(range(len(boxes)), 2):
        extent = np.minimum(boxes[i][1], boxes[j][1])-np.maximum(boxes[i][0], boxes[j][0])
        if np.all(extent > 1e-10):
            penetrations.append(dict(boxes=[i, j], overlap_volume_m3=float(np.prod(extent))))
    return dict(vertices=len(v), referenced_vertices=len(used), triangles=len(f), bounds=mesh.bounds.tolist(),
                finite=bool(np.isfinite(v).all()), indices_valid=bool(f.min() >= 0 and f.max() < len(v)),
                degenerate_triangles=int((mesh.area_faces < 1e-14).sum()),
                winding_consistent=bool(mesh.is_winding_consistent), watertight=bool(mesh.is_watertight),
                components=[dict(vertices=len(x.vertices), triangles=len(x.faces), watertight=bool(x.is_watertight),
                                 signed_volume_m3=float(x.volume) if x.is_watertight else None,
                                 bounds=x.bounds.tolist()) for x in components],
                box_components=len(boxes), box_volume_penetrations=penetrations,
                welded_boundary_edges=int((edge_counts == 1).sum()),
                welded_nonmanifold_edges=int((edge_counts > 2).sum()),
                exact_duplicate_triangles=sum(n-1 for n in dup),
                positive_area_coplanar_triangle_pairs=overlap,
                vertices_sharing_nonparallel_face_normals=hard_vertices,
                normal_axis_counts={str(tuple(map(float, n))): int(c)
                                    for n, c in Counter(map(tuple, np.round(mesh.face_normals, 9))).items()})


def continuous_exterior(mesh):
    """Diagnostic exterior of axis-aligned box assemblies plus open border planes.

    Split existing planes on original coordinate breakpoints, omit patches whose
    outward side is inside a closed box, and retain each remaining patch once.
    No displacement, rounding below 1e-12 m only, no dependency on Boolean engines.
    This diagnostic copy is NEVER a substitute for the collision asset.
    """
    triangles = np.round(mesh.triangles, 12)
    coords = [np.unique(triangles[..., i]) for i in range(3)]
    boxes = [np.round(x.bounds, 12) for x in mesh.split(only_watertight=False)
             if x.is_watertight and len(x.vertices) == 8 and len(x.faces) == 12]
    groups = defaultdict(list)
    for tri, normal in zip(triangles, mesh.face_normals):
        axis = int(np.argmax(np.abs(normal)))
        if abs(normal[axis]) < 1-1e-10:
            raise ValueError('Diagnostic exterior supports axis-aligned box faces only')
        groups[(axis, float(tri[0, axis]), int(np.sign(normal[axis])))].append(tri)
    vertices, faces, lookup = [], [], {}
    for (axis, position, sign), tris in groups.items():
        other = [k for k in range(3) if k != axis]
        projected = np.asarray(tris)[..., other]
        for u0, u1 in zip(coords[other[0]][:-1], coords[other[0]][1:]):
            for v0, v1 in zip(coords[other[1]][:-1], coords[other[1]][1:]):
                uv = np.array([(u0+u1)/2, (v0+v1)/2])
                covered = False
                for tri in projected:
                    a, b, c = tri
                    e, f, d = b-a, c-a, uv-a
                    det = np.linalg.det(np.stack((e, f), axis=1))
                    if abs(det) < 1e-14:
                        continue
                    s, t = np.linalg.solve(np.stack((e, f), axis=1), d)
                    if s >= -1e-10 and t >= -1e-10 and s+t <= 1+1e-10:
                        covered = True; break
                if not covered:
                    continue
                outward = np.zeros(3); outward[axis] = position + sign*1e-7; outward[other] = uv
                if any(np.all(outward > b[0]+1e-10) and np.all(outward < b[1]-1e-10) for b in boxes):
                    continue
                quad = np.zeros((4, 3)); quad[:, axis] = position
                quad[:, other] = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
                if np.cross(quad[1]-quad[0], quad[2]-quad[0])[axis]*sign < 0:
                    quad = quad[[0, 3, 2, 1]]
                ids = []
                for point in quad:
                    key = tuple(point)
                    if key not in lookup:
                        lookup[key] = len(vertices); vertices.append(point)
                    ids.append(lookup[key])
                faces.extend([[ids[0], ids[1], ids[2]], [ids[0], ids[2], ids[3]]])
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def compare_surfaces(original, candidate, asset):
    with np.load(Path(asset)/'height_map.npz') as data:
        xx, yy = np.meshgrid(data['x_m'], data['y_m'])
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        stored = data['height_m'].ravel()
    a, b = top_surface(original, xy), top_surface(candidate, xy)
    valid = np.isfinite(a) & np.isfinite(b)
    samples = np.vstack((candidate.vertices, candidate.triangles_center))
    _, distance, _ = trimesh.proximity.closest_point(original, samples)
    return dict(heightmap_cell_centers=len(xy), same_coverage=bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
                max_top_height_difference_m=float(np.max(np.abs(a[valid]-b[valid]))),
                original_vs_exported_heightmap_max_m=float(np.max(np.abs(a[valid]-stored[valid]))),
                candidate_samples_on_original_surface_max_m=float(distance.max()),
                original_bounds=original.bounds.tolist(), candidate_bounds=candidate.bounds.tolist())
