import os
import math
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from plot_utils import save_and_plot_metrics

DATA_DIR = "/work/pi_kaleel_mahmood_uri_edu/syed_uri/LLM_DATASETS"
TOKENIZER_ID = "Qwen/Qwen2.5-0.5B"
BLOCK_SIZE = 2048

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--model_name", type=str, default="qwen2.5_scratch")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = args.data_dir

    model_dir = os.path.join(args.output_dir, args.model_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, cache_dir=args.data_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=BLOCK_SIZE,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config)

    raw_data = load_dataset("wikitext", "wikitext-103-v1", cache_dir=args.data_dir)
    tokenized = raw_data.map(lambda x: tokenizer(x["text"]), batched=True, remove_columns=["text"], num_proc=4)

    def chunk_tokens(examples):
        concat = {k: sum(examples[k], []) for k in examples.keys()}
        total = (len(concat["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
        res = {k: [t[i : i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)] for k, t in concat.items()}
        res["labels"] = res["input_ids"].copy()
        return res

    lm_dataset = tokenized.map(chunk_tokens, batched=True, num_proc=4)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=300,
        save_steps=300,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=args.lr,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=25,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
                early_stopping_threshold=0.001
            )
        ]
    )

    trainer.train()

    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    metrics = trainer.evaluate(lm_dataset["test"])
    final_ppl = math.exp(metrics["eval_loss"])
    print(f"\nFinal Test Perplexity: {final_ppl:.2f}\n")

    save_and_plot_metrics(
        log_history=trainer.state.log_history,
        base_output_dir=args.output_dir,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()
