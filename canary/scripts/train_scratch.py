"""Qwen2.5 (768-dim, 12 layers) from scratch on WikiText-103: torchrun -m canary.scripts.train_scratch

Logs train_log.csv and eval_log.csv (validation perplexity) during training; metrics.csv and plots at the end.
"""
import argparse
import csv
import math
import os

import torch

from ..config import DATA_DIR, OUTPUT_ROOT
from ..data.wikitext import load, prepare_lm_datasets
from ..models.qwen import csv_logger


def save_and_plot_metrics(log_history, target_dir, model_name):
    train = [(e["step"], e["loss"]) for e in log_history if "step" in e and "loss" in e]
    evals = [(e["step"], e["eval_loss"], math.exp(e["eval_loss"])) for e in log_history if "step" in e and "eval_loss" in e]
    with open(os.path.join(target_dir, "metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["Type", "Step", "Loss", "Perplexity"]] +
                                [["Train", s, f"{l:.4f}", f"{math.exp(l):.2f}"] for s, l in train] +
                                [["Eval", s, f"{l:.4f}", f"{p:.2f}"] for s, l, p in evals])
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} Training Progress", fontsize=14, fontweight="bold")
    if train:
        ax1.plot(*zip(*train), label="Train Loss", color="#1f77b4", alpha=0.75, linewidth=1.5)
    if evals:
        steps, losses, ppls = zip(*evals)
        ax1.plot(steps, losses, label="Validation Loss", color="#d62728", marker="o", markersize=4, linewidth=2)
        ax2.plot(steps, ppls, label="Validation PPL", color="#2ca02c", marker="s", markersize=4, linewidth=2)
        ax2.set(xlabel="Steps", ylabel="Perplexity", title="Validation Perplexity")
        best, at = min(ppls), steps[ppls.index(min(ppls))]
        span = max(ppls) - best
        ax2.annotate(f"Best: {best:.2f}", xy=(at, best), xytext=(at, best + (span * 0.08 if span > 0 else best * 0.05)),
                     arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4), fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()
    ax1.set(xlabel="Steps", ylabel="Loss", title="Training & Validation Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(os.path.join(target_dir, f"training_metrics.{ext}"), **({"dpi": 150} if ext == "png" else {}))
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Pretrain Qwen2.5 from scratch on WikiText-103")
    for name, default in dict(data_dir=DATA_DIR, output_dir=OUTPUT_ROOT, model_name="qwen2.5_scratch",
                              tokenizer_id="Qwen/Qwen2.5-0.5B", block_size=2048, batch_size=4, grad_accum=8, epochs=10,
                              lr=4e-4, warmup_ratio=0.03, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.95,
                              eval_steps=300, save_steps=300, num_proc=4, patience=6,
                              early_stopping_threshold=0.001).items():
        p.add_argument(f"--{name}", type=type(default), default=default)
    p.add_argument("--use_early_stopping", action="store_true")
    a = p.parse_args()
    from transformers import (AutoTokenizer, EarlyStoppingCallback, Qwen2Config, Qwen2ForCausalLM, Trainer,
                              TrainingArguments, default_data_collator)

    os.makedirs(a.data_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = a.data_dir
    model_dir = os.path.join(a.output_dir, a.model_name)
    os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer_id, cache_dir=a.data_dir)
    except OSError:
        import glob
        snaps = sorted(glob.glob(os.path.join(a.data_dir, "models--" + a.tokenizer_id.replace("/", "--"), "snapshots", "*")))
        if not snaps:
            raise
        tokenizer = AutoTokenizer.from_pretrained(snaps[-1], local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=768, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=12,
        num_key_value_heads=4, max_position_embeddings=a.block_size, rope_theta=1000000.0, rms_norm_eps=1e-6,
        tie_word_embeddings=True, sliding_window=None, use_sliding_window=False))
    raw = load(a.data_dir)
    data = prepare_lm_datasets(raw, tokenizer, a.block_size, a.num_proc)
    args = TrainingArguments(
        output_dir=os.path.join(model_dir, "checkpoints"), eval_strategy="steps",
        eval_steps=a.eval_steps, save_steps=a.save_steps, save_total_limit=2,
        load_best_model_at_end=a.use_early_stopping,
        metric_for_best_model="eval_loss", greater_is_better=False, learning_rate=a.lr, weight_decay=a.weight_decay,
        adam_beta1=a.adam_beta1, adam_beta2=a.adam_beta2, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
        warmup_steps=a.warmup_ratio, lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), logging_steps=25,
        dataloader_num_workers=4, ddp_find_unused_parameters=False, report_to="none")
    callbacks = [csv_logger(model_dir)] + ([EarlyStoppingCallback(
        early_stopping_patience=a.patience, early_stopping_threshold=a.early_stopping_threshold)] if a.use_early_stopping else [])
    trainer = Trainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["validation"],
                      data_collator=default_data_collator, callbacks=callbacks)
    trainer.train()
    if not a.use_early_stopping:
        trainer.save_model(os.path.join(model_dir, "last"))
        if trainer.state.best_model_checkpoint is not None:
            trainer._load_best_model()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    test_loss = trainer.evaluate(data["test"])["eval_loss"]
    words = sum(len(t.split()) for t in raw["test"]["text"] if t.strip())
    tokens_per_word = len(data["test"]) * a.block_size / max(words, 1)
    print(f"\nTEST: loss {test_loss:.4f} | token ppl {math.exp(test_loss):.2f} | tokens/word {tokens_per_word:.3f} | "
          f"word ppl {math.exp(test_loss * tokens_per_word):.2f}\n")
    save_and_plot_metrics(trainer.state.log_history, model_dir, a.model_name)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
