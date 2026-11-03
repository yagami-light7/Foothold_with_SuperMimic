"""Select top-level training references without importing CUDA or the simulator."""
from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path


def select_reference_files(motion_dir, clips=None):
    root = Path(motion_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {root}")
    available = {p.stem: p for p in sorted(root.glob('*.csv')) if p.is_file()}
    if not available:
        raise FileNotFoundError(f"No training CSVs in {root}; generate references first")
    selected = set()
    for pattern in clips or ['*']:
        if not pattern or '/' in pattern or '\\' in pattern:
            raise ValueError(f"Use clip names or quoted glob patterns, not paths: {pattern!r}")
        pattern = pattern.removesuffix('.csv')
        matches = [name for name in available if fnmatch.fnmatchcase(name, pattern)]
        if not matches:
            raise FileNotFoundError(f"No reference matches {pattern!r} in {root}")
        selected.update(matches)
    files = [available[name] for name in sorted(selected)]
    for path in files:
        if not path.with_suffix('.json').is_file():
            raise FileNotFoundError(f"Missing reference metadata: {path.with_suffix('.json')}")
    return [str(path) for path in files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion_dir', required=True)
    parser.add_argument('--clips', nargs='+')
    args = parser.parse_args()
    try:
        files = select_reference_files(args.motion_dir, args.clips)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Selected {len(files)} clips: ' + ', '.join(Path(f).stem for f in files))


if __name__ == '__main__':
    main()
