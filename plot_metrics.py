"""Plot one or more columns of a CSV metrics log against an x column and save a PNG.

    python plot_metrics.py --csv output/ecp24_pg19/train_log.csv --x iteration --y train_loss \
        --out output/ecp24_pg19/train_loss.png
    python plot_metrics.py --csv output/ecp24_pg19/eval_log.csv --x iteration --y test_ppl validation_ppl \
        --out output/ecp24_pg19/perplexity.png --log_y

Empty, NaN and infinite values are skipped. Points are sorted by x.
"""
import argparse
import csv
import math
import os


def plot_metrics(csv_path, x, ys, out_png, title=None, log_y=False):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    missing = [c for c in [x, *ys] if c not in rows[0]]
    if missing:
        raise KeyError(f"columns not found in {csv_path}: {missing}")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        for y in ys:
            pts = [(float(r[x]), float(r[y])) for r in rows if r[x] != "" and r[y] != ""]
            pts = sorted((a, b) for a, b in pts if math.isfinite(a) and math.isfinite(b))
            if pts:
                xs, vs = zip(*pts)
                ax.plot(xs, vs, label=y, marker="o" if len(pts) <= 50 else None, markersize=3)
        ax.set_xlabel(x)
        ax.set_ylabel(ys[0] if len(ys) == 1 else "value")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(title or os.path.splitext(os.path.basename(out_png))[0])
        ax.grid(True, linestyle="--", alpha=0.5)
        if len(ys) > 1:
            ax.legend()
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.savefig(out_png, dpi=150)
    finally:
        plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description="Plot CSV metric columns to a PNG.")
    p.add_argument("--csv", required=True)
    p.add_argument("--x", required=True, help="x column, e.g. iteration or step")
    p.add_argument("--y", nargs="+", required=True, help="one or more y columns")
    p.add_argument("--out", required=True, help="output .png path")
    p.add_argument("--title", default=None)
    p.add_argument("--log_y", action="store_true")
    a = p.parse_args()
    print("wrote", plot_metrics(a.csv, a.x, a.y, a.out, a.title, a.log_y))


if __name__ == "__main__":
    main()
