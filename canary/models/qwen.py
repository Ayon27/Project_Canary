"""Qwen2 trained from scratch on a canary token stream (HF Trainer)."""
import csv
import math
import os

import numpy as np
import torch

from ..data.wikitext import load, prepare_lm_datasets
from ..utils import out, tokenizer


class Blocks(torch.utils.data.Dataset):
    def __init__(self, file, block):
        self.a, self.block = np.load(file, mmap_mode="r"), block

    def __len__(self):
        return len(self.a) // self.block  # final partial block dropped

    def __getitem__(self, i):
        x = torch.from_numpy(self.a[i * self.block:(i + 1) * self.block].astype(np.int64))
        return {"input_ids": x, "labels": x}


def csv_logger(d):
    """Trainer callback: appends train loss to d/train_log.csv and validation loss and perplexity to d/eval_log.csv
    as they are logged."""
    from transformers import TrainerCallback

    def append(name, row):
        with open(os.path.join(d, name), "a", newline="") as f:
            csv.writer(f).writerow(row)

    class CSVLogger(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            if state.is_world_process_zero:
                for name, cols in (("train_log.csv", ["step", "train_loss"]),
                                   ("eval_log.csv", ["step", "validation_loss", "validation_ppl"])):
                    with open(os.path.join(d, name), "w", newline="") as f:
                        csv.writer(f).writerow(cols)

        def on_log(self, args, state, control, logs=None, **kw):
            if state.is_world_process_zero and logs:
                if "loss" in logs:
                    append("train_log.csv", [state.global_step, logs["loss"]])
                if "eval_loss" in logs:
                    append("eval_log.csv", [state.global_step, logs["eval_loss"], math.exp(logs["eval_loss"])])

    return CSVLogger()


def train(c, m):
    from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments, default_data_collator, set_seed

    tok, d, B, seed, epochs = tokenizer(c["tokenizer"]), out(c, "models", m, ""), c["block"], c["train_seed"], c["epochs"]
    set_seed(seed)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tok), **c["arch"], max_position_embeddings=B, rope_theta=1e6, rms_norm_eps=1e-6,
        tie_word_embeddings=True, sliding_window=None, use_sliding_window=False))
    cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=d, num_train_epochs=epochs, per_device_train_batch_size=c["batch_size"],
        per_device_eval_batch_size=c["batch_size"], gradient_accumulation_steps=c["grad_accum"],
        learning_rate=4e-4, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01, adam_beta1=0.9,
        adam_beta2=0.95, optim="adamw_torch_fused" if cuda else "adamw_torch",
        bf16=cuda and torch.cuda.is_bf16_supported(), tf32=cuda and torch.cuda.get_device_capability()[0] >= 8,
        eval_strategy="steps", eval_steps=300, save_strategy="steps" if epochs > 1 else "no", save_steps=300,
        save_total_limit=2, load_best_model_at_end=epochs > 1, metric_for_best_model="eval_loss",
        greater_is_better=False, logging_steps=25, dataloader_num_workers=4, seed=seed, data_seed=seed,
        report_to="none")
    val = prepare_lm_datasets({"validation": load()["validation"]}, tok, B, num_proc=1)["validation"]
    trainer = Trainer(model=model, args=args, train_dataset=Blocks(out(c, "corpora", f"{m}_stream.npy"), B),
                      eval_dataset=val, data_collator=default_data_collator, callbacks=[csv_logger(d)])
    trainer.train()
    trainer.evaluate()  # final validation row, after the best model is loaded
    trainer.save_model(d)  # model.safetensors marks the stage as done


def scorer(c, m):
    """Judge settings: pad with EOS to the longest row; lm_head on the scored positions only."""
    from transformers import Qwen2ForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2ForCausalLM.from_pretrained(out(c, "models", m, ""), torch_dtype=torch.float32).to(device).eval()
    return dict(feats=lambda x: model.model(input_ids=x).last_hidden_state, head=model.lm_head,
                pad=tokenizer(c["tokenizer"]).eos_token_id, width=None, chunk=25, device=model.device)
