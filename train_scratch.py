import os
import math
import argparse
import itertools
import re
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
)
from plot_utils import save_and_plot_metrics

DATA_DIR = "/work/pi_kaleel_mahmood_uri_edu/syed_uri/LLM_DATASETS"
TOKENIZER_ID = "Qwen/Qwen2.5-0.5B"
BLOCK_SIZE = 2048


def prepare_lm_datasets(raw_dataset, tokenizer, block_size, num_proc=4):
    """
    1. Injects EOS token at document (article) boundaries so the model learns document ends.
    2. Tokenizes text without intra-batch truncation.
    3. Concatenates all tokens across each split continuously and chunks into exact block_size.
       Guarantees at most ONE partial chunk (< block_size tokens) is dropped per split,
       preserving >99.99% of training data and >99.2% of evaluation data.
    """
    lm_splits = {}
    is_article_start = lambda s: bool(re.match(r"^\s*=\s+[^=]+\s+=\s*$", s))
    eos = tokenizer.eos_token or "<|endoftext|>"

    for split in ["train", "validation", "test"]:
        if split not in raw_dataset:
            continue
        print(f"\n[DataPrep] Formatting and tokenizing '{split}' split...")
        raw_texts = raw_dataset[split]["text"]

        formatted_texts = []
        for i, text in enumerate(raw_texts):
            if not text.strip():
                continue
            if is_article_start(text) and i > 0:
                formatted_texts.append(eos + "\n" + text)
            else:
                formatted_texts.append(text)

        temp_ds = Dataset.from_dict({"text": formatted_texts})
        tokenized_ds = temp_ds.map(
            lambda x: tokenizer(x["text"], add_special_tokens=False),
            batched=True,
            remove_columns=["text"],
            num_proc=num_proc,
            desc=f"Tokenizing {split}",
        )

        flat_ids = list(itertools.chain.from_iterable(tokenized_ds["input_ids"]))
        total_tokens = len(flat_ids)
        trimmed_len = (total_tokens // block_size) * block_size
        remainder = total_tokens - trimmed_len
        flat_ids = flat_ids[:trimmed_len]

        blocks = [flat_ids[i : i + block_size] for i in range(0, trimmed_len, block_size)]
        pct_kept = (trimmed_len / max(total_tokens, 1)) * 100
        print(
            f"[DataPrep] '{split}' tokens: {total_tokens:,} | Blocks (len {block_size}): {len(blocks):,} | "
            f"Remainder dropped: {remainder} ({100 - pct_kept:.3f}%) | Retained: {pct_kept:.3f}%"
        )

        lm_splits[split] = Dataset.from_dict({
            "input_ids": blocks,
            "labels": [b.copy() for b in blocks],
        })

    return lm_splits


def main():
    parser = argparse.ArgumentParser(description="Pretrain Qwen2.5 from scratch on WikiText-103")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to HF datasets cache")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base output directory")
    parser.add_argument("--model_name", type=str, default="qwen2.5_scratch", help="Experiment / model name")
    parser.add_argument("--tokenizer_id", type=str, default=TOKENIZER_ID, help="HF tokenizer repo or path")
    parser.add_argument("--block_size", type=int, default=BLOCK_SIZE, help="Context sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for cosine scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="AdamW beta2 (0.95 recommended for LLMs)")
    parser.add_argument("--eval_steps", type=int, default=300, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=300, help="Save checkpoint every N steps")
    parser.add_argument("--num_proc", type=int, default=4, help="Preprocessing workers")
    parser.add_argument("--use_early_stopping", action="store_true", default=False, help="Enable early stopping callback")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (eval rounds)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001, help="Min improvement required")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = args.data_dir

    model_dir = os.path.join(args.output_dir, args.model_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, cache_dir=args.data_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=args.block_size,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        sliding_window=None,
        use_sliding_window=False,
    )
    model = Qwen2ForCausalLM(config)

    raw_data = load_dataset("wikitext", "wikitext-103-v1", cache_dir=args.data_dir)
    lm_dataset = prepare_lm_datasets(
        raw_dataset=raw_data,
        tokenizer=tokenizer,
        block_size=args.block_size,
        num_proc=args.num_proc,
    )

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=25,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    callbacks = []
    if args.use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    trainer.train()

    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    metrics = trainer.evaluate(lm_dataset["test"])
    test_loss = metrics["eval_loss"]
    token_ppl = math.exp(test_loss)

    total_test_words = sum(len(text.split()) for text in raw_data["test"]["text"] if text.strip())
    total_test_tokens = len(lm_dataset["test"]) * args.block_size
    tokens_per_word = total_test_tokens / max(total_test_words, 1)
    word_ppl = math.exp(test_loss * tokens_per_word)

    print("\n" + "=" * 55)
    print("FINAL TEST EVALUATION METRICS:")
    print(f"  Test Cross-Entropy Loss:   {test_loss:.4f}")
    print(f"  Token-Level Perplexity:    {token_ppl:.2f}")
    print(f"  Token-to-Word Ratio:       {tokens_per_word:.3f}")
    print(f"  Estimated Word Perplexity: {word_ppl:.2f}")
    print("=" * 55 + "\n")

    save_and_plot_metrics(
        log_history=trainer.state.log_history,
        base_output_dir=args.output_dir,
        model_name=args.model_name,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
