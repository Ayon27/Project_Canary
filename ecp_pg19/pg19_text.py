"""PG-19 text -> bert-base-cased token ids, using the authors' WikiText rule from WikitextDataset.py.

WikitextDataset (ECPTransformer-master/WikitextDataset.py) does, for every line from f.readlines():
    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
        line.strip(' ').replace('\\n', '[SEP]').replace('<unk>', '[UNK]')))
and concatenates the ids of all lines. We apply exactly that to PG-19.

Books are processed one at a time and concatenated in file-list order. This equals running the authors' rule
on a single text file made by concatenating the books, provided every book ends with a newline; a book whose
last line has no trailing newline gets one (`read_book_lines`). The file is opened in text mode with
encoding="utf-8", as in WikitextDataset (universal newlines: "\\r\\n" and "\\r" become "\\n").
"""
import numpy as np

from ecp_pg19.ecp_base import ORIGINAL_TOKENIZER_KWARGS, ORIGINAL_TOKENIZER_NAME

TOKEN_DTYPE = np.uint16  # bert-base-cased ids are < 28996 < 65536


def load_tokenizer(cache_dir=None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ORIGINAL_TOKENIZER_NAME, cache_dir=cache_dir, **ORIGINAL_TOKENIZER_KWARGS)


def transform_line(line: str) -> str:
    """The per-line string transform of WikitextDataset."""
    return line.strip(' ').replace('\n', '[SEP]').replace('<unk>', '[UNK]')


def read_book_lines(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    return lines


def tokenize_lines(tokenizer, lines) -> np.ndarray:
    ids = []
    for line in lines:
        ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(transform_line(line))))
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= np.iinfo(TOKEN_DTYPE).max):
        raise ValueError(f"token id out of uint16 range: min={arr.min()} max={arr.max()}")
    return arr.astype(TOKEN_DTYPE)


def tokenize_book(tokenizer, path: str) -> np.ndarray:
    return tokenize_lines(tokenizer, read_book_lines(path))
