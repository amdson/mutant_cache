"""Optional W&B tracking for supervised experiments; imported without the SDK."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import uuid
import warnings


def add_tracking_args(parser):
    group = parser.add_argument_group("experiment tracking")
    group.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    group.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "mutant_cache"))
    group.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    group.add_argument("--wandb-group", default=None)
    group.add_argument("--wandb-name", default=None)
    group.add_argument("--wandb-tags", nargs="*", default=[])
    group.add_argument("--wandb-log-every", type=int, default=50,
                       help="optimizer steps between training updates (validation is always logged)")


def finite_scalars(values):
    """Omit undefined evaluation metrics, including the initial NaN loss."""
    return {k: v for k, v in values.items()
            if isinstance(v, (int, float)) and math.isfinite(v)}


def load_tracking_config(path):
    config = json.loads(Path(path).read_text())
    allowed = {"wandb_mode", "wandb_project", "wandb_entity", "wandb_group", "wandb_name",
               "wandb_tags", "wandb_log_every"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("tracking config must be an object containing only supported wandb_* settings")
    if config.get("wandb_mode", "disabled") not in ("disabled", "online", "offline"):
        raise ValueError("invalid wandb_mode in tracking config")
    interval = config.get("wandb_log_every", 50)
    if not isinstance(interval, int) or interval <= 0:
        raise ValueError("wandb_log_every must be a positive integer")
    if "wandb_tags" in config and (not isinstance(config["wandb_tags"], list)
            or any(not isinstance(tag, str) for tag in config["wandb_tags"])):
        raise ValueError("wandb_tags must be a list of strings")
    return config


class ExperimentTracker:
    def __init__(self, args, run_dir):
        self.args = args
        self.run_dir = Path(run_dir)
        self.mode = getattr(args, "wandb_mode", "disabled")
        self.run = None
        self.broken = False
        self.started = time.perf_counter()
        self.record = {}

    def _write_record(self, **updates):
        self.record.update(updates)
        path = self.run_dir / "tracking.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.record, indent=2))
        temporary.replace(path)

    def __enter__(self):
        if self.mode == "disabled":
            return self
        if getattr(self.args, "wandb_log_every", 50) <= 0:
            raise ValueError("--wandb-log-every must be positive")
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B tracking requires the optional SDK: pip install -e '.[tracking]'") from exc
        context = getattr(self.args, "experiment_context", {})
        group = getattr(self.args, "wandb_group", None) or context.get("sweep")
        name = getattr(self.args, "wandb_name", None)
        if not name:
            trial = context.get("trial_id", getattr(self.args, "tag", None) or self.run_dir.name)
            name = f"{trial}-s{self.args.seed}"
            if context.get("attempt"):
                name += f"-attempt{context['attempt']}"
        tags = list(getattr(self.args, "wandb_tags", []))
        if context.get("stage"):
            tags.append(f"stage-{context['stage']}")
        # A retry is a fresh training run, not a resumed trajectory. Each call
        # has its own ID; the ID and URL remain discoverable in local artifacts.
        self.run = wandb.init(project=getattr(self.args, "wandb_project", "mutant_cache"),
            entity=getattr(self.args, "wandb_entity", None), mode=self.mode,
            group=group, name=name, tags=tags, id=uuid.uuid4().hex[:16],
            job_type="training", dir=str(self.run_dir),
            config=json.loads(json.dumps(vars(self.args), default=str)),
            save_code=False, settings={"disable_git": True, "console": "off", "init_timeout": 30})
        try:
            self.run.define_metric("trainer/step")
            self.run.define_metric("*", step_metric="trainer/step")
            self._write_record(mode=self.mode, id=self.run.id, name=self.run.name,
                project=self.run.project, entity=self.run.entity, group=group,
                url=self.run.url if self.mode == "online" else None,
                local_dir=self.run.dir, sdk_version=wandb.__version__, status="running")
        except BaseException:
            self.run.finish(exit_code=1)
            raise
        return self

    @property
    def interval(self):
        return getattr(self.args, "wandb_log_every", 50) if self.run is not None else 0

    def _send(self, method, *args, **kwargs):
        if self.run is None or self.broken:
            return
        try:
            method(*args, **kwargs)
        except Exception as exc:
            # The local history/checkpoints remain authoritative if the
            # external logger fails after training has started.
            self.broken = True
            warnings.warn(f"W&B logging stopped; training continues: {exc}", RuntimeWarning)
            self._write_record(logging_error=str(exc))

    def log(self, values, step):
        if self.run is not None:
            self._send(self.run.log, {"trainer/step": step, **finite_scalars(values)})

    def progress(self, step, total, rows_seen, loss, phase, optimizer_groups):
        values = {"train/loss_running": loss, "progress/fraction": step / max(total, 1),
            "progress/rows_seen": rows_seen, "progress/elapsed_s": time.perf_counter() - self.started,
            "progress/singles_phase": int(phase == "singles")}
        values.update({f"lr/{g.get('name', i)}": g["lr"] for i, g in enumerate(optimizer_groups)})
        self.log(values, step)

    def validation(self, row):
        values = {"progress/exposure": row["exposure"], "train/loss": row["train_loss"],
                  "train/lr_scale": row["lr_scale"]}
        values.update({"val/" + k[4:]: v for k, v in row.items() if k.startswith("val_")})
        self.log(values, row["step"])

    def summary(self, values):
        if self.run is not None:
            self._send(self.run.summary.update, values)

    def selections(self, selections):
        self.summary({f"selection/{rule}/step": record["step"]
                      for rule, record in selections.items()})

    def complete(self, extra):
        values = {"selected/step": extra["selection"].get("step"),
            "selected/rule": extra["selection"]["rule"], "model/active_params": extra["active_params"],
            "model/head_params": extra["active_head_params"]}
        for row in extra["val_summary"]:
            for aggregation in ("macro", "pooled"):
                value = row.get(aggregation)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values[f"selected/val/{row['depth']}_{row['metric']}_{aggregation}"] = value
        for prefix in ("timing", "memory"):
            values.update({prefix + "/" + k: v for k, v in finite_scalars(extra[prefix]).items()})
        self.summary(values)

    def __exit__(self, exc_type, exc, tb):
        if self.run is None:
            return False
        status = "failed" if exc_type else "finished"
        # Finish even after a logging failure, to release the SDK process.
        try:
            self.run.summary.update({"training/status": status})
        except Exception as tracking_error:
            warnings.warn(f"W&B status update failed: {tracking_error}", RuntimeWarning)
        try:
            self.run.finish(exit_code=1 if exc_type else 0)
        except Exception as tracking_error:
            warnings.warn(f"W&B finalization failed: {tracking_error}", RuntimeWarning)
            self._write_record(finalization_error=str(tracking_error))
        self._write_record(status=status)
        return False
