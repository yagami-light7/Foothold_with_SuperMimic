"""Import Roban assets and retire the old data/ layout without changing file contents.

The external source is read-only. Existing destination files must match exactly.
Run from any directory with the standard Python interpreter; no simulator needed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / 'library'
DEFAULT_SOURCE = ROOT.parent / 'third_party/library'
ROBAN_DATASETS = ('kimodo_fallgetup', 'lafan1', 'snapmogen_roban_50hz', 'style100_roban_50hz')
S54_MOVES = {
    'data/s54_description': 'library/model/kuavo_s54',
    'data/s54_newton_usd': 'library/model/kuavo_s54/usd',
    'data/retarget_data': 'library/motion_reference/kuavo_s54/retarget_data',
    'data/s54_flat_dataset': 'library/motion_reference/kuavo_s54/datasets/s54_flat_dataset',
    'data/s54_stair_dataset': 'library/motion_reference/kuavo_s54/datasets/s54_stair_dataset',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def copy_exact(source, target):
    expected = sha256(source)
    if target.exists():
        if sha256(target) != expected:
            raise FileExistsError(f'Different destination file; refusing to overwrite: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.importing')
        shutil.copy2(source, temporary)
        if sha256(temporary) != expected:
            raise IOError(f'Copy checksum mismatch: {source}')
        temporary.replace(target)
    return dict(path=str(target.relative_to(LIBRARY)), bytes=source.stat().st_size, sha256=expected)


def import_roban(source, workers):
    model = source / 'model/roban_s22_0901'
    if not (model / 'xml/biped_s17_verified_handball_fixedhead.xml').is_file():
        raise FileNotFoundError(f'Roban model not found: {model}')
    pairs = [(p, LIBRARY / 'model/roban_s22_0901' / p.relative_to(model))
             for p in sorted(model.rglob('*')) if p.is_file()]
    for dataset in ROBAN_DATASETS:
        folder = source / 'motion_reference/roban' / dataset
        files = sorted(folder.rglob('*.csv'))
        if not files:
            raise FileNotFoundError(f'No retargeted CSVs: {folder}')
        pairs.extend((p, LIBRARY / 'motion_reference/roban' / dataset / p.relative_to(folder)) for p in files)
        log = folder / 'retarget.log'
        if log.exists():
            pairs.append((log, LIBRARY / 'motion_reference/roban' / dataset / log.name))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def transfer(pair):
            src, dst = pair
            row = copy_exact(src, dst)
            row['source_path'] = str(src.relative_to(source))
            return row
        for row in pool.map(transfer, pairs):
            rows.append(row)
            if len(rows) % 1000 == 0:
                print(f'Roban: {len(rows)}/{len(pairs)} files verified', flush=True)
    report = dict(status='completed', source_root=str(source),
                  updated_at=datetime.now(timezone.utc).isoformat(), source_modified=False,
                  selection='complete Roban model; all retargeted CSVs and generation logs',
                  excluded=['kimodo_fall_getup_bvh: raw BVH retained at the external source'],
                  files=len(rows), bytes=sum(r['bytes'] for r in rows), entries=rows)
    write_json(LIBRARY / 'manifests/roban_import.json', report)
    print(f'Roban import complete: {len(rows)} files, {report["bytes"] / 1e9:.3f} GB', flush=True)


def migrate_data():
    manifest = LIBRARY / 'manifests/data_migration.json'
    if manifest.exists():
        report = json.loads(manifest.read_text())
        rows = report['entries']
    else:
        mapping = {**S54_MOVES, **{
            f'library/motion_reference/{kind}': f'library/motion_reference/kuavo_s54/{kind}'
            for kind in ('bvh', 'csv', 'pkl', 'npz')}}
        rows = []
        for old, new in mapping.items():
            source, target = ROOT / old, ROOT / new
            if not source.exists():
                continue
            for p in sorted(source.rglob('*')):
                if p.is_file():
                    rows.append(dict(old=str(p.relative_to(ROOT)),
                                     new=str((target / p.relative_to(source)).relative_to(ROOT)),
                                     bytes=p.stat().st_size, sha256=sha256(p)))
        known = {r['old'] for r in rows}
        unknown = [str(p.relative_to(ROOT)) for p in (ROOT / 'data').rglob('*')
                   if p.is_file() and str(p.relative_to(ROOT)) not in known]
        if unknown:
            raise ValueError(f'Unmapped old data files; nothing moved: {unknown}')
        report = dict(status='planned', entries=rows, files=len(rows),
                      bytes=sum(r['bytes'] for r in rows),
                      created_at=datetime.now(timezone.utc).isoformat())
    # Preflight all collisions before deleting even an identical duplicate.
    for row in rows:
        source, target = ROOT / row['old'], ROOT / row['new']
        for path in (source, target):
            if path.exists() and sha256(path) != row['sha256']:
                raise FileExistsError(f'Changed file; refusing migration: {path}')
        if not source.exists() and not target.exists():
            raise FileNotFoundError(row['old'])
    write_json(manifest, report)
    for row in rows:
        source, target = ROOT / row['old'], ROOT / row['new']
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                source.unlink()  # Byte-identical duplicate; checked above.
            else:
                source.rename(target)
        if sha256(target) != row['sha256']:
            raise IOError(f'Migration checksum mismatch: {target}')
    for folder in [ROOT / 'data', *[LIBRARY / 'motion_reference' / k for k in ('bvh', 'csv', 'pkl', 'npz')]]:
        if folder.exists():
            for d in sorted((p for p in folder.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                d.rmdir()
            folder.rmdir()
    report.update(status='completed', original_file_contents_preserved=True,
                  completed_at=datetime.now(timezone.utc).isoformat())
    write_json(manifest, report)
    print(f'Old layout retired: {len(rows)} files moved or deduplicated, all checksums verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--migrate-only', action='store_true')
    parser.add_argument('--import-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or (args.migrate_only and args.import_only):
        parser.error('Positive worker count and one operation selection required')
    if not args.migrate_only:
        import_roban(args.source.resolve(), args.workers)
    if not args.import_only:
        migrate_data()


if __name__ == '__main__':
    main()
