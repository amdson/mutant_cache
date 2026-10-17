"""
mutant_cache/eval/table.py
--------------------------
The results table (PLAN.md 26), with two rules baked in so they cannot be
forgotten at writing-up time:

1. **Zero-shot entries live in a separate, labelled context block.**
   DATASETS.md: "One comparison to avoid: the zero-shot leaderboard. VenusREM
   at ρ = 0.523 and VenusREM2 at 0.556 see no DMS labels, so placing a
   supervised number beside them is a category error."  ``kind="zero_shot"``
   rows are never interleaved with supervised rows, and the block carries the
   reason in its header.

2. **Every correlation prints its reliability ceiling.**  A cell is rendered
   ``0.612 (≤0.88)`` and, where the ceiling is known, the attainment ratio
   ``ρ/ceiling`` is available as its own column.  A row with no known ceiling
   renders ``(≤ ?)`` rather than nothing, so an unknown ceiling is visible
   instead of silently reading as 1.0.

Correlation columns are detected by name (``spearman``, ``pearson``, ``rho``,
``r``, ``eps_*``, anything prefixed ``depth_``); pass ``correlation_cols`` to
override.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("model", "ablation", "baseline", "control", "zero_shot", "published")

BLOCKS = [
    ("model", "Model"),
    ("ablation", "Ablations — what each tie would mean is in DATASETS.md"),
    ("baseline", "Supervised baselines"),
    ("control", "Controls"),
    ("published", "Published supervised comparators"),
    ("zero_shot", "Context only — these methods see no DMS labels; "
                  "they are NOT comparators for a supervised number"),
]

_CORR_NAMES = {"spearman", "pearson", "rho", "r", "kendall", "auc",
               "spearman_over_ceiling"}


def _is_corr(name: str) -> bool:
    n = name.lower()
    return (n in _CORR_NAMES or n.startswith("depth_") or n.startswith("eps_sp")
            or n.startswith("eps_pe") or n.endswith("_spearman")
            or n.endswith("_pearson") or n.endswith("_rho"))


@dataclass
class ResultRow:
    name: str
    kind: str = "model"
    metrics: dict = field(default_factory=dict)
    ceiling: dict | float | None = None
    n: int | None = None
    note: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    def ceiling_for(self, metric: str) -> float:
        if self.ceiling is None:
            return float("nan")
        if isinstance(self.ceiling, dict):
            return float(self.ceiling.get(metric, float("nan")))
        return float(self.ceiling)


class ResultsTable:
    """Collect rows, render markdown / csv / json."""

    def __init__(self, title: str = "Results", correlation_cols=None):
        self.title = title
        self.rows: list[ResultRow] = []
        self.correlation_cols = set(correlation_cols) if correlation_cols else None

    # -- building ------------------------------------------------------------
    def add(self, name, metrics, *, kind="model", ceiling=None, n=None, note=""):
        self.rows.append(ResultRow(name=name, kind=kind, metrics=dict(metrics),
                                   ceiling=ceiling, n=n, note=note))
        return self

    def add_zero_shot(self, name, metrics, *, ceiling=None, n=None, note=""):
        """Convenience: DATASETS.md's context row, never a competitor."""
        return self.add(name, metrics, kind="zero_shot", ceiling=ceiling, n=n,
                        note=note or "no labels seen")

    def add_depth_table(self, name, depth_table: pd.DataFrame, *, kind="model",
                        group_col="score_set_id", note=""):
        """Fold a ``depth_spearman`` ALL block into one row.

        The ceiling comes from the same frame, so the per-depth reliability
        printed beside each correlation is the one that produced it.
        """
        a = depth_table[depth_table[group_col] == "ALL"]
        metrics, ceil = {}, {}
        for r in a.itertuples():
            metrics[f"depth_{r.depth}"] = float(r.spearman)
            ceil[f"depth_{r.depth}"] = float(r.reliability)
        n = int(a[a["depth"] == "all"]["n"].iloc[0]) if len(a) else None
        return self.add(name, metrics, kind=kind, ceiling=ceil, n=n, note=note)

    def add_throughput(self, name, *, variants_per_s_cache, variants_per_s_full,
                       accuracy_delta=None, note=""):
        """DATASETS.md's compute baseline: throughput ratio + accuracy delta."""
        m = {"variants_per_s_cache": variants_per_s_cache,
             "variants_per_s_full": variants_per_s_full,
             "throughput_ratio": (variants_per_s_cache / variants_per_s_full
                                  if variants_per_s_full else float("nan"))}
        if accuracy_delta is not None:
            m["accuracy_delta"] = accuracy_delta
        return self.add(name, m, kind="control", note=note)

    # -- rendering -----------------------------------------------------------
    def metric_names(self) -> list[str]:
        seen: list[str] = []
        for r in self.rows:
            for k in r.metrics:
                if k not in seen:
                    seen.append(k)
        return seen

    def _corr(self, name: str) -> bool:
        if self.correlation_cols is not None:
            return name in self.correlation_cols
        return _is_corr(name)

    def to_frame(self) -> pd.DataFrame:
        """Long form — one row per (entry, metric), with the ceiling alongside."""
        recs = []
        for r in self.rows:
            for k, v in r.metrics.items():
                c = r.ceiling_for(k) if self._corr(k) else float("nan")
                recs.append({"name": r.name, "kind": r.kind, "metric": k,
                             "value": float(v) if v is not None else np.nan,
                             "is_correlation": self._corr(k), "ceiling": c,
                             "attainment": (float(v) / c
                                            if self._corr(k) and c == c and c
                                            else np.nan),
                             "n": r.n, "note": r.note})
        return pd.DataFrame(recs)

    def to_markdown(self, float_fmt: str = "{:.3f}") -> str:
        cols = self.metric_names()
        head = "| entry | n | " + " | ".join(cols) + " | note |"
        sep = "|---" * (len(cols) + 3) + "|"
        lines = [f"## {self.title}", "",
                 "Correlations print as `value (≤ceiling)`; the ceiling is the "
                 "assay's reliability, and `?` means it was never estimated.",
                 "", head, sep]
        for kind, label in BLOCKS:
            block = [r for r in self.rows if r.kind == kind]
            if not block:
                continue
            lines.append(f"| **{label}** |" + " |" * (len(cols) + 2))
            for r in block:
                cells = []
                for c in cols:
                    v = r.metrics.get(c)
                    if v is None:
                        cells.append("—")
                        continue
                    s = float_fmt.format(v)
                    if self._corr(c):
                        ce = r.ceiling_for(c)
                        s += f" (≤{float_fmt.format(ce)})" if ce == ce else " (≤ ?)"
                    cells.append(s)
                n = "" if r.n is None else f"{r.n:,}"
                lines.append(f"| {r.name} | {n} | " + " | ".join(cells)
                             + f" | {r.note} |")
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_markdown()

    def write(self, path: str | Path) -> Path:
        """Write ``<stem>.md``, ``<stem>.csv`` and ``<stem>.json`` together."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        stem = p.with_suffix("")
        stem.with_suffix(".md").write_text(self.to_markdown())
        self.to_frame().to_csv(stem.with_suffix(".csv"), index=False)
        stem.with_suffix(".json").write_text(json.dumps(
            {"title": self.title,
             "rows": [{"name": r.name, "kind": r.kind, "metrics": r.metrics,
                       "ceiling": r.ceiling, "n": r.n, "note": r.note}
                      for r in self.rows]}, indent=2, default=float))
        return stem.with_suffix(".md")


def write_results_table(path, title, rows) -> Path:
    """``rows`` = iterable of dicts accepted by :meth:`ResultsTable.add`."""
    t = ResultsTable(title)
    for r in rows:
        t.add(r.pop("name"), r.pop("metrics"), **r)
    return t.write(path)
