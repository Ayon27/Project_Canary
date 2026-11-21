"""Stage report: judge_raw.csv (raw table) and detection_counts.csv (the 3x3 T0 table + the null number)."""
import csv
from collections import Counter

from .common import ALPHA, GROUPS, K, MODELS, WORKS_PER_GROUP, path
from .judge import COLUMNS


def run():
    rows = []
    for m in MODELS:
        with open(path("judge", f"{m}.csv"), newline="") as f:
            rows += list(csv.reader(f))[1:]
    assert len(rows) == len(MODELS) * len(GROUPS) * WORKS_PER_GROUP * K
    with open(path("report", "judge_raw.csv"), "w", newline="") as f:
        csv.writer(f).writerows([COLUMNS] + rows)

    # A work is "trained on" when its released canary has the lowest loss of all K (rank 1, i.e. p <= ALPHA).
    detected, total = Counter(), Counter()
    for m, fmt, r, _, _, released, _, _, rank in rows:
        if released == "1":
            total[m] += 1
            detected[(m, fmt, int(r))] += int(rank) / K <= ALPHA

    # detection_counts.csv: the 3 x 3 table of T0 detections out of 50, then the one null number out of 450.
    formats = list(dict.fromkeys(f for f, _ in GROUPS))
    reps = sorted({r for _, r in GROUPS})
    with open(path("report", "detection_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"T0 detected / {WORKS_PER_GROUP}"] + [f"repetition={r}" for r in reps])
        for fmt in formats:
            w.writerow([fmt] + [detected[("T0", fmt, r)] for r in reps])
        w.writerow([])
        w.writerow([f"null detected / {total['null']}", sum(detected[("null", fmt, r)] for fmt in formats for r in reps)])
