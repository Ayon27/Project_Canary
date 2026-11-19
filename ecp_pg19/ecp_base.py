"""Read-only access to the original ECP code in ../ECPTransformer-master.

Nothing in ECPTransformer-master is modified. This module:
  * puts ECPTransformer-master on sys.path and imports the original model classes;
  * extracts `configure_optimizers` and `compute_perplexity_huggingface` from ECPTransformer_main.py
    with the `ast` module and executes only those two function definitions. The file cannot be imported
    normally because it calls parser.parse_args() and downloads a tokenizer at import time.
"""
import ast
import math
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ECP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ECPTransformer-master"))
if not os.path.isdir(ECP_ROOT):
    raise ImportError(f"ECPTransformer-master not found at {ECP_ROOT}")
if ECP_ROOT not in sys.path:
    sys.path.insert(0, ECP_ROOT)


@contextmanager
def no_bytecode():
    """Import modules from ECPTransformer-master without writing __pycache__ files into that folder."""
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = old


with no_bytecode():
    from models.ECPAttentionTransformer import ECPAttentionTransformer  # noqa: E402  (original, unmodified)
    from models.AutoRegressiveWrapper import AutoRegressiveWrapper  # noqa: E402  (original, unmodified)

ORIGINAL_TOKENIZER_NAME = "bert-base-cased"
ORIGINAL_TOKENIZER_KWARGS = dict(truncation=True, max_length=512)  # as in ECPTransformer_main.py
BERT_BASE_CASED_VOCAB = 28996  # NUM_UNIQUE_TOKENS in ECPTransformer_main.py for word-level modeling

_MAIN_PATH = os.path.join(ECP_ROOT, "ECPTransformer_main.py")


def _load_main_functions(names, args_ns):
    """Execute only the named top-level function definitions of ECPTransformer_main.py.

    `args_ns` plays the role of the module-level `args` object those functions read.
    Returns the namespace that holds the functions (and the LAST_BEST_PERPLEXITY global).
    """
    with open(_MAIN_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=_MAIN_PATH)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {d.name for d in defs}
    if missing:
        raise RuntimeError(f"functions not found in {_MAIN_PATH}: {sorted(missing)}")
    module = ast.Module(body=defs, type_ignores=[])
    ns = {
        "__name__": "ecp_main_extracted",
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
        "math": math,
        "args": args_ns,
        "LAST_BEST_PERPLEXITY": 999,
    }
    exec(compile(module, _MAIN_PATH, "exec"), ns)
    return ns


def authors_configure_optimizers(model, lr):
    """The authors' AdamW setup: betas (0.9, 0.95), wd 0.1 on Linear weights, 0 on biases/LayerNorm/Embedding.

    Returns an optimizer with exactly two param groups: [0] = decay, [1] = no_decay.
    """
    ns = _load_main_functions(["configure_optimizers"], SimpleNamespace(lr=lr))
    return ns["configure_optimizers"](model)


def authors_compute_perplexity(model, test_set, device, sequence_length, net_dropout):
    """The authors' perplexity: overlapping (sequence_length+1)-token windows, stride sequence_length//2,
    ppl = exp(mean over windows of the window's mean token loss). `test_set.data` must be a 2D LongTensor.

    `model` must be the unwrapped AutoRegressiveWrapper (the function accesses model.model.layers).
    Returns (ppl as float, last window loss as float).
    """
    args_ns = SimpleNamespace(sequence_length=sequence_length, net_dropout=net_dropout)
    ns = _load_main_functions(["compute_perplexity_huggingface"], args_ns)
    _, ppl, loss = ns["compute_perplexity_huggingface"](model, test_set, device, max_len=sequence_length)
    return float(ppl), float(loss)
