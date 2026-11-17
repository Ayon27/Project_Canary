import os
import math
import csv
import re

def save_and_plot_metrics(log_history, base_output_dir="./output", model_name="qwen2.5_scratch"):
    safe_name = re.sub(r'[^\w\-_\.]', '_', model_name.strip()).strip('_')
    target_dir = os.path.join(base_output_dir, safe_name)
    os.makedirs(target_dir, exist_ok=True)

    csv_file = os.path.join(target_dir, "metrics.csv")
    plot_file = os.path.join(target_dir, "training_metrics.png")

    train_steps, train_losses = [], []
    eval_steps, eval_losses, eval_ppls = [], [], []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])
            eval_ppls.append(math.exp(entry["eval_loss"]))

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Step", "Loss", "Perplexity"])
        for s, l in zip(train_steps, train_losses):
            writer.writerow(["Train", s, f"{l:.4f}", f"{math.exp(l):.2f}"])
        for s, l, p in zip(eval_steps, eval_losses, eval_ppls):
            writer.writerow(["Eval", s, f"{l:.4f}", f"{p:.2f}"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{model_name} Training Progress", fontsize=14, fontweight="bold")

        if train_steps:
            ax1.plot(train_steps, train_losses, label="Train Loss", color="#1f77b4", alpha=0.75, linewidth=1.5)
        if eval_steps:
            ax1.plot(eval_steps, eval_losses, label="Validation Loss", color="#d62728", marker="o", markersize=4, linewidth=2)
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        if eval_ppls:
            ax2.plot(eval_steps, eval_ppls, label="Validation PPL", color="#2ca02c", marker="s", markersize=4, linewidth=2)
            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Perplexity")
            ax2.set_title("Validation Perplexity")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend()

            min_ppl = min(eval_ppls)
            min_step = eval_steps[eval_ppls.index(min_ppl)]
            ax2.annotate(
                f"Best: {min_ppl:.2f}",
                xy=(min_step, min_ppl),
                xytext=(min_step, min_ppl * 1.05),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
                fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(plot_file, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plotting error: {e}")

    return {
        "target_dir": target_dir,
        "csv_path": csv_file,
        "plot_path": plot_file,
        "best_eval_ppl": min(eval_ppls) if eval_ppls else None
    }
