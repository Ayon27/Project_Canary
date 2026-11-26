"""Experiment defaults. Every key can be overridden on the command line, e.g. --epochs 3."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env(path):
    env = {}
    if not os.path.isfile(path):
        return env
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_ENV = _load_env(os.path.join(ROOT, ".env"))
DATA_DIR = _ENV.get("DATA_DIR", os.path.join(ROOT, "data"))
OUTPUT_ROOT = _ENV.get("OUTPUT_DIR", os.path.join(ROOT, "output"))
LOGS_DIR = _ENV.get("LOGS_DIR", os.path.join(ROOT, "logs"))
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", DATA_DIR)

MODELS = ("T0", "null")  # trained with and without canaries

CANARY = dict(K=100, alpha=0.01, block=2048, works_per_group=50, n_heldout=450,
              groups=[(f, r) for f in ("rand_alnum", "rand_words", "carrier") for r in (1, 5, 10)],
              min_sentences=10, length_margin=64, min_leak_tokens=20,
              selection_seed=1000, group_seed=1001, insertion_seed=1002, train_seed=42)

QWEN = dict(model="qwen", tokenizer="Qwen/Qwen2.5-0.5B", eos="<|endoftext|>", dtype="int32", exact_decode=True,
            arch=dict(hidden_size=896, intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14,
                      num_key_value_heads=2),
            epochs=10, batch_size=2, grad_accum=16)

# ECP-24 (214,646,596 parameters). Global batch 16 = authors' batch 8 x 2 accumulation steps.
ECP24 = dict(model="ecp", tokenizer="bert-base-cased", eos="[SEP]", dtype="uint16", exact_decode=False,
             layers=24, dim=768, heads=6, sequence_length=2048, num_segments=8, net_dropout=0.05, vocab_size=28996,
             precision="bf16", lr=2e-4, lr_halving_every=100_000, global_batch=16, original_grad_accum=2,
             micro_batch=4, order_seed=0, log_every=100, validate_every=5000, eval_batch=8)

EXPERIMENTS = {
    "qwen_wikitext": {**CANARY, **QWEN, "dataset": "wikitext"},
    "ecp_wikitext": {**CANARY, **ECP24, "dataset": "wikitext", "num_iterations": 200_000, "save_every": 5000,
                     "exact_resume": False},
    # num_iterations None: 20 tokens per parameter. exact_resume: save RNG states, check config on resume.
    "ecp_pg19": {**ECP24, "dataset": "pg19", "data_dir": f"{DATA_DIR}/pg19/tokens", "train_seed": 0,
                 "num_iterations": None, "tokens_per_param": 20.0, "save_every": 2000, "exact_resume": True},
}
