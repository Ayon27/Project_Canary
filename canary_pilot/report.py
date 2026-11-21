"""Stage report: judge_raw.csv and detection_counts.csv"""
import csv
from collections import Counter

from scipy.stats import beta

from .common import ALPHA, GROUPS, K, MODELS, WORKS_PER_GROUP, path
from .judge import COLUMNS


def cp(k, n):
    return (beta.ppf(0.025, k, n - k + 1) if k else 0.0), (beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def run():
    rows = []
    for m in MODELS:
        with open(path("judge", f"{m}.csv"), newline="") as f:
            rows += list(csv.reader(f))[1:]
    assert len(rows) == len(MODELS) * len(GROUPS) * WORKS_PER_GROUP * K
    with open(path("report", "judge_raw.csv"), "w", newline="") as f:
        csv.writer(f).writerows([COLUMNS] + rows)

    # Detections (p <= ALPHA) per (model, format, repetition): the 9 T0 cells out of 50, and null out of 450.
    hits, n = Counter(), Counter()
    for m, fmt, r, _, _, released, _, _, rank in rows:
        if released == "1":
            for key in ((m, fmt, r), (m, "all", "all")):
                n[key] += 1
                hits[key] += int(rank) / K <= ALPHA
    with open(path("report", "detection_counts.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["model", "format", "repetition", "detected", "n", "ci95_low", "ci95_high"]] +
                                [[*key, hits[key], n[key], *(f"{x:.4f}" for x in cp(hits[key], n[key]))] for key in n])
