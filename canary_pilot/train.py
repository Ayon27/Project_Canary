"""Stage train: train_scratch.py's Qwen model and recipe from scratch on one stream, 1 epoch, final weights."""

import csv
import math

import numpy as np
import torch

from .common import BLOCK, TRAIN_SEED, load_wikitext, path, tokenizer


class Blocks(torch.utils.data.Dataset):
    def __init__(self, file):
        self.a = np.load(file, mmap_mode="r")

    def __len__(self):
        return len(self.a) // BLOCK  # the final partial block is dropped

    def __getitem__(self, i):
        x = torch.from_numpy(self.a[i * BLOCK:(i + 1) * BLOCK].astype(np.int64))
        return {"input_ids": x, "labels": x}


def run(model_name):
    from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments, default_data_collator, set_seed
    from train_scratch import prepare_lm_datasets

    tok, out = tokenizer(), path("models", model_name, "")
    set_seed(TRAIN_SEED)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tok), hidden_size=768, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=12,
        num_key_value_heads=4, max_position_embeddings=BLOCK, rope_theta=1e6, rms_norm_eps=1e-6,
        tie_word_embeddings=True, sliding_window=None, use_sliding_window=False))
    args = TrainingArguments(
        output_dir=out, num_train_epochs=1, per_device_train_batch_size=4, per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, learning_rate=4e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.95, optim="adamw_torch_fused", bf16=True, tf32=True,
        eval_strategy="steps", eval_steps=300, save_strategy="no", logging_steps=25, dataloader_num_workers=4,
        seed=TRAIN_SEED, data_seed=TRAIN_SEED, report_to="none")
    val = prepare_lm_datasets({"validation": load_wikitext()["validation"]}, tok, BLOCK, num_proc=1)["validation"]
    trainer = Trainer(model=model, args=args, train_dataset=Blocks(path("corpora", f"{model_name}_stream.npy")),
                      eval_dataset=val, data_collator=default_data_collator)
    trainer.train()
    trainer.evaluate()
    hist = trainer.state.log_history  # loss logs for plot_metrics.py
    with open(out + "train_log.csv", "w", newline="") as f:
        csv.writer(f).writerows([["step", "train_loss"]] + [[h["step"], h["loss"]] for h in hist if "loss" in h])
    with open(out + "eval_log.csv", "w", newline="") as f:
        csv.writer(f).writerows([["step", "validation_loss", "validation_ppl"]] +
                                [[h["step"], h["eval_loss"], math.exp(h["eval_loss"])] for h in hist if "eval_loss" in h])
    trainer.save_model(out)  # model.safetensors, written last: marks the stage as done
