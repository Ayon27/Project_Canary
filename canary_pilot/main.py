"""Orchestrator. From the repository root:

    python -m canary_pilot.main                          # every stage whose output is missing, in order
    python -m canary_pilot.main train judge report --force   # rerun these stages (retrain on the existing data)
    python -m canary_pilot.main train --models null      # train / judge one model only

Stages: works -> corpora -> train -> judge -> report. A stage is skipped when its output exists, unless --force.
"""
import argparse
import os

from . import corpora, judge, report, train, works
from .common import MODELS, path

STAGES = {  # stage: (function, per model, output that marks it as done)
    "works": (works.run, False, lambda m: path("works", "works_manifest.jsonl")),
    "corpora": (corpora.run, False, lambda m: path("corpora", "T0_stream.npy")),
    "train": (train.run, True, lambda m: path("models", m, "model.safetensors")),
    "judge": (judge.run, True, lambda m: path("judge", f"{m}.csv")),
    "report": (report.run, False, lambda m: path("report", "detection_counts.csv")),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", help=f"any of: {' '.join(STAGES)} (default: all)")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--force", action="store_true", help="rerun even if the output exists")
    args = ap.parse_args()
    if set(args.stages) - set(STAGES):
        ap.error(f"unknown stage(s) {set(args.stages) - set(STAGES)}")
    for name, (fn, per_model, output) in STAGES.items():
        for m in args.models if per_model else [None]:
            if name in (args.stages or STAGES) and (args.force or not os.path.exists(output(m))):
                print(f"== {name} {m or ''}", flush=True)
                fn(m) if per_model else fn()


if __name__ == "__main__":
    main()
