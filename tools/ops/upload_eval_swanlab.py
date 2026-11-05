"""Append an audited Eval/ scalar bundle to an existing SwanLab experiment.

Dry-run by default. A successful upload requires cloud read-back verification;
the training CSV, its uploader cursor, and existing training metrics are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


TRAINING_KEYS = ["Reward/EpisodeReturn", "Reward/Simulator"]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def canonical(response):
    return {
        item["key"]: {int(point["index"]): float(point["data"])
                      for point in item.get("metrics", [])}
        for item in response.get("list", [])
    }


def validate(bundle):
    assert bundle["target"]["run_id"] and bundle["target"]["name"]
    rows = bundle["rows"]
    assert rows and [r["step"] for r in rows] == sorted({r["step"] for r in rows})
    keys = set(rows[0]["metrics"])
    assert keys and all(k.startswith("Eval/") for k in keys)
    for row in rows:
        assert isinstance(row["step"], int) and row["step"] >= 0
        assert set(row["metrics"]) == keys
        assert all(isinstance(v, (int, float)) and math.isfinite(v)
                   for v in row["metrics"].values())
    return sorted(keys)


def check_values(actual, rows):
    missing = []
    for row in rows:
        for key, expected in row["metrics"].items():
            observed = actual.get(key, {}).get(row["step"])
            if observed is None or not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-8):
                missing.append([key, row["step"], expected, observed])
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    keys = validate(bundle)
    print(json.dumps(dict(target=bundle["target"], steps=[r["step"] for r in bundle["rows"]],
                          metrics_per_checkpoint=len(keys), total_scalar_points=len(keys)*len(bundle["rows"])),
                     ensure_ascii=False), flush=True)
    if not (args.execute or args.verify_only):
        print("Dry-run passed; use --execute to upload.", flush=True)
        return

    import swanlab
    target = bundle["target"]
    directory = args.bundle.resolve().parent
    status_path = directory / "upload_status.json"
    status = dict(status="starting", target=target, bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest())

    def record(state, **extra):
        status.update(status=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(status_path, status)

    record("checking_cloud")
    try:
        api = swanlab.Api()
        experiment = api.run(f'{target["workspace"]}/{target["project"]}/{target["run_id"]}')
        # The public URL/run slug and server's internal cuid can differ.
        assert experiment.run_id and experiment.name == target["name"], "Wrong experiment or unavailable cloud run"
        record("checking_cloud", cloud_run_id=experiment.run_id, url=experiment.url)
        before = canonical(experiment.metrics(keys=TRAINING_KEYS, sample=1500, ignore_timestamp=True))
        assert all(before.get(k) for k in TRAINING_KEYS), "Original training curves were not readable"
        previous = directory / "cloud_training_before.json"
        if previous.exists():
            baseline = {k: {int(s): v for s, v in points.items()}
                        for k, points in json.loads(previous.read_text()).items()}
            assert baseline == before, "Training curves changed since first upload attempt"
        else:
            write_json(previous, before)
        existing = canonical(experiment.metrics(keys=keys, sample=1500, ignore_timestamp=True))
        if not args.verify_only:
            pending = []
            for row in bundle["rows"]:
                values = {}
                for key, value in row["metrics"].items():
                    old = existing.get(key, {})
                    if row["step"] in old:
                        assert math.isclose(old[row["step"]], value, rel_tol=1e-6, abs_tol=1e-8), "Conflicting existing Eval point"
                    else:
                        assert not old or row["step"] > max(old), "SDK cannot backfill below this metric's existing last step"
                        values[key] = value
                if values:
                    pending.append((row["step"], values))
            if pending:
                swanlab.merge_settings(swanlab.Settings(interactive=False, probe=swanlab.Settings.Probe(monitor=False)))
                config = json.loads(Path(bundle["training_args_path"]).read_text())
                record("uploading", pending_scalar_points=sum(len(v) for _, v in pending))
                swanlab.init(project=target["project"], workspace=target["workspace"], id=target["run_id"],
                             resume="must", name=target["name"], config=config,
                             log_dir=str(directory / "swanlab"), mode="online")
                try:
                    for step, values in pending:
                        swanlab.log(values, step=step)
                        print(f"Logged {len(values)} Eval scalars at checkpoint {step}", flush=True)
                finally:
                    # This job supplements a completed training experiment. Its
                    # local status records upload failures without relabeling
                    # the original completed training as crashed.
                    swanlab.finish(state="success")

        record("verifying_cloud")
        for attempt in range(6):
            observed = canonical(experiment.metrics(keys=keys, sample=1500, ignore_timestamp=True))
            missing = check_values(observed, bundle["rows"])
            if not missing:
                break
            print(f"Cloud read-back attempt {attempt+1}: {len(missing)} points pending", flush=True)
            if attempt < 5:
                time.sleep(5)
        write_json(directory / "cloud_eval_readback.json", observed)
        assert not missing, f"Cloud verification failed for {len(missing)} points; first: {missing[:1]}"
        after = canonical(experiment.metrics(keys=TRAINING_KEYS, sample=1500, ignore_timestamp=True))
        assert before == after, "Original training curves changed"
        record("verified", scalar_points=len(keys)*len(bundle["rows"]), original_training_curves_unchanged=True)
        print(f"VERIFIED: {len(keys)*len(bundle['rows'])} Eval points; training curves unchanged; {experiment.url}", flush=True)
    except BaseException as error:
        record("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
