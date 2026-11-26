"""Stage report: judge_raw.csv, detection_counts.csv (detected when rank / K <= alpha), and mink_plus_plus_summary.csv."""
import csv
import os
from collections import Counter

from ..config import MODELS
from ..utils import out
from .judge import COLUMNS
from .mink_plus_plus import METRIC_COLS


def run(c):
    rows = []
    for m in MODELS:
        with open(out(c, "judge", f"{m}.csv"), newline="") as f:
            rows += list(csv.reader(f))[1:]
    assert len(rows) == len(MODELS) * len(c["groups"]) * c["works_per_group"] * c["K"]
    with open(out(c, "report", "judge_raw.csv"), "w", newline="") as f:
        csv.writer(f).writerows([COLUMNS] + rows)

    detected, total = Counter(), Counter()
    for m, fmt, r, _, _, released, _, _, rank in rows:
        if released == "1":
            total[m] += 1
            detected[(m, fmt, int(r))] += int(rank) / c["K"] <= c["alpha"]
    formats, reps = list(dict.fromkeys(f for f, _ in c["groups"])), sorted({r for _, r in c["groups"]})
    with open(out(c, "report", "detection_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"T0 detected / {c['works_per_group']}"] + [f"repetition={r}" for r in reps])
        w.writerows([fmt] + [detected[("T0", fmt, r)] for r in reps] for fmt in formats)
        w.writerow([])
        w.writerow([f"null detected / {total['null']}", sum(detected[("null", f, r)] for f in formats for r in reps)])

    mink_files = [out(c, "mink_plus_plus", f"{m}_metrics.csv") for m in MODELS]
    if all(os.path.exists(p) for p in mink_files):
        mink_rows = []
        for p in mink_files:
            with open(p, newline="") as f:
                mink_rows += list(csv.reader(f))[1:]
        summary = out(c, "report", "mink_plus_plus_summary.csv")
        with open(summary + ".tmp", "w", newline="") as f:
            csv.writer(f).writerows([METRIC_COLS] + mink_rows)
        os.replace(summary + ".tmp", summary)
