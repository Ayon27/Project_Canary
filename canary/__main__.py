"""Run experiment stages from the repository root. Every config key is a flag (see -h after the experiment name).

    python -m canary qwen_wikitext --epochs 10
    python -m canary ecp_wikitext train --models T0 --num_iterations 200000 --name ecp_test
    torchrun --standalone --nproc_per_node=4 -m canary ecp_pg19 train --name ecp24_pg19

Outputs go to output/<name> (default: the experiment name). A stage is skipped when its output exists, unless --force.
Each run appends its full config to output/<name>/run_config.jsonl.
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import time

import numpy as np

from .config import EXPERIMENTS, MODELS, OUTPUT_ROOT, ROOT
from .data import wikitext
from .models import ecp, qwen
from .pipeline import corpora, judge, report, works
from .utils import out, plot_run, read_json


def train(c, m=None):
    if c["model"] == "qwen":
        return qwen.train(c, m)
    if c["dataset"] == "pg19":
        d = c["data_dir"]
        return ecp.train(c, np.memmap(f"{d}/train.uint16.bin", np.uint16, mode="r"), d, c["out"],
                         sha=read_json(f"{d}/train.manifest.json")["sha256"])
    ecp.train(c, np.load(out(c, "corpora", f"{m}_stream.npy"), mmap_mode="r"), out(c, "eval_data"),
              out(c, "models", m), prepare=lambda: wikitext.prepare_ecp_eval(c))


def stages(c):
    """name: (function, per model, done marker or None to always run). ECP training resumes from last.pt."""
    if c["dataset"] == "pg19":
        return {"train": (train, False, None),
                "eval": (ecp.evaluate, False, lambda m: out(c, "best.pt.eval_fp32.json")),
                "plot": (plot_run, False, None)}
    qwen_done = lambda m: out(c, "models", m, "model.safetensors")
    return {"works": (works.run, False, lambda m: out(c, "works", "works_manifest.jsonl")),
            "corpora": (corpora.run, False, lambda m: out(c, "corpora", "T0_stream.npy")),
            "train": (train, True, qwen_done if c["model"] == "qwen" else None),
            "judge": (judge.run, True, lambda m: out(c, "judge", f"{m}.csv")),
            "report": (report.run, False, lambda m: out(c, "report", "detection_counts.csv")),
            "plot": (plot_run, False, None)}


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("experiment", choices=EXPERIMENTS)
    exp = pre.parse_known_args()[0].experiment
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=EXPERIMENTS)
    ap.add_argument("stages", nargs="*", help="default: all")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--name", default=exp, help="output folder under output/")
    ap.add_argument("--force", action="store_true", help="rerun even if the output exists")
    g = ap.add_argument_group(f"{exp} config")
    for k, v in EXPERIMENTS[exp].items():
        if k not in ("model", "dataset"):
            typ = type(v) if isinstance(v, (int, float, str)) and not isinstance(v, bool) else ast.literal_eval
            g.add_argument(f"--{k}", type=typ, default=v, help=f"default: {v}")
    a = ap.parse_args()
    c = {k: getattr(a, k, v) for k, v in EXPERIMENTS[exp].items()}
    return a, dict(c, out=os.path.join(OUTPUT_ROOT, a.name))


def record(c, a, stage_names):
    """Append the command, git commit and full config to run_config.jsonl (rank 0 only)."""
    if os.environ.get("RANK", "0") != "0":
        return
    git = lambda *cmd: subprocess.run(["git", "-C", ROOT, *cmd], capture_output=True, text=True).stdout.strip()
    os.makedirs(c["out"], exist_ok=True)
    with open(os.path.join(c["out"], "run_config.jsonl"), "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv[1:], stages=stage_names,
                                models=a.models, git_commit=git("rev-parse", "HEAD"),
                                git_dirty=bool(git("status", "--porcelain")), config=c)) + "\n")


def main():
    a, c = parse_args()
    st = stages(c)
    if set(a.stages) - set(st):
        sys.exit(f"unknown stage(s) {set(a.stages) - set(st)}; choose from {list(st)}")
    record(c, a, a.stages or list(st))
    for name, (fn, per_model, done) in st.items():
        for m in a.models if per_model else [None]:
            if name in (a.stages or st) and (a.force or done is None or not os.path.exists(done(m))):
                print(f"== {name} {m or ''}", flush=True)
                fn(c, m) if per_model else fn(c)


if __name__ == "__main__":
    main()
