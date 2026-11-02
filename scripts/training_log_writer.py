"""Keep TensorBoard logs and stream complete PPO iterations to the supervisor."""

import json
import time
from pathlib import Path

from rsl_rl.utils.log_writer import LogWriter
from torch.utils.tensorboard import SummaryWriter


class TensorboardJsonWriter(LogWriter):
    def __init__(self, log_dir: str):
        self.tensorboard = SummaryWriter(log_dir=log_dir, flush_secs=10)
        self.stream = (Path(log_dir) / "metrics.jsonl").open("a", buffering=1)
        self.step: int | None = None
        self.values: dict[str, float] = {}
        self.closed = False

    def _flush(self) -> None:
        if self.step is not None and self.values:
            self.stream.write(json.dumps({"iteration": self.step, "wall_time": time.time(),
                                          "metrics": self.values}) + "\n")
            self.values = {}

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        value = float(scalar_value)
        self.tensorboard.add_scalar(tag, value, global_step)
        # RSL also emits wall-time-indexed duplicates; these use a different axis.
        if tag.endswith("/time"):
            return
        if self.step != global_step:
            self._flush()
            self.step = global_step
        self.values[tag] = value

    def stop(self) -> None:
        if not self.closed:
            self._flush()
            self.stream.close()
            self.tensorboard.close()
            self.closed = True
