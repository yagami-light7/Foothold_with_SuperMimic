"""Offline Kit-less MJCF→USD import. Run in the existing supermimic image, not Kit."""
import argparse
import hashlib
import json
import sys
import shutil
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.s54 import MJCF_PATH, USD_DIR, USD_PATH, checkpoint_contract


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=MJCF_PATH)
    ap.add_argument('--output',type=Path,default=USD_DIR)
    a=ap.parse_args()
    if not a.source.is_file(): raise FileNotFoundError('Run python tools/prepare_s54_asset.py first')
    from pxr import Plug
    import isaacsim.asset.importer.utils.impl as iu
    Plug.Registry().RegisterPlugins(str(Path(iu.__file__).parent / 'schemas'))
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
    # Never silently substitute the old 27-DoF URDF-derived USD.
    expected=a.output / a.source.stem / (a.source.stem+'.usda')
    from check_s54_usd import check
    # The importer silently adds _1/_2 when its output exists. Import in a fresh
    # directory and validate its returned path, then publish to the configured path.
    # Otherwise a rerun can validate yesterday's USD instead of the new conversion.
    with tempfile.TemporaryDirectory(prefix='s54_usd_') as staging:
        cfg=MJCFImporterConfig(mjcf_path=str(a.source.resolve()),usd_path=staging)
        actual=Path(MJCFImporter(cfg).import_mjcf())
        if not actual.is_file(): raise FileNotFoundError(f'Importer returned missing USD: {actual}')
        staged_report=check(actual)
        if not staged_report['pass']: raise RuntimeError(f'Imported USD alignment failed: {staged_report}')
        expected.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(actual.parent,expected.parent,dirs_exist_ok=True)
    print(expected)
    report=check(expected)
    (a.output/'alignment.json').write_text(json.dumps(report,indent=2))
    if not report['pass']: raise RuntimeError('USD alignment failed; see alignment.json')
    (a.output/'contract.json').write_text(json.dumps(dict(**checkpoint_contract(),
        sim_mjcf_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(), usd=str(expected)),indent=2))
    print('PASS static USD alignment. Run tools/test_env_s54.py for backend roundtrip/FK.')


if __name__ == '__main__': main()
