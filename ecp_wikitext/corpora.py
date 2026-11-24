"""Stage corpora: T0 and null token streams using BERT tokenizer."""
import itertools
import random

import numpy as np

from .common import BLOCK, EOS, INSERTION_SEED, candidate_tokens, encode, path, read_jsonl, train_articles, write_json


def encode_background(articles):
    """Layout: each row tokenized, EOS + '\\n' in front of the title row."""
    out = []
    for i in range(0, len(articles), 256):
        chunk = articles[i:i + 256]
        ids = encode([EOS + "\n" + r if j == 0 else r for a in chunk for j, r in enumerate(a)])
        ends = np.cumsum([len(a) for a in chunk])
        out += [np.fromiter(itertools.chain(*ids[e - len(a):e]), np.uint16) for a, e in zip(chunk, ends)]
    return out


def encode_work(w):
    """T0 and null documents; the canary is t0[s:e]."""
    t0, null = encode([EOS + "\n" + w["t0_text"] + "\n", EOS + "\n" + w["null_text"] + "\n"])
    p, (tail,) = candidate_tokens(w["judge_prefix"], [w["released"]])
    s, e = len(p), len(p) + len(tail)
    assert t0[:e] == p + tail and t0[:s] + t0[e:] == null, w["work_id"]
    return np.array(t0, np.uint16), np.array(null, np.uint16), s, e


def null_hits(texts, works):
    """Candidates that occur as words in texts."""
    keys = {tuple(c[:-1].lower().split()) if w["format"] == "rand_words" else (c[:-1].split()[-1].lower(),)
            for w in works for c in w["candidates"]}
    first, hits = {k[0] for k in keys}, []
    for t in texts:
        ws = t.lower().split()
        hits += [t for i, x in enumerate(ws) if x in first and ({tuple(ws[i:i + 1]), tuple(ws[i:i + 8])} & keys)]
    return hits


def run():
    works, held = read_jsonl(path("works", "works_manifest.jsonl")), read_jsonl(path("works", "heldout_works.jsonl"))
    articles = train_articles()
    excluded = {w["article_index"] for w in works + held}
    bg_rows = [a for k, a in enumerate(articles) if k not in excluded]
    bg = encode_background(bg_rows)
    docs = [encode_work(w) for w in works]

    rng = random.Random(INSERTION_SEED)
    copies = sorted(((rng.randint(0, len(bg)), w["work_id"]) for w in works for _ in range(w["repetition"])),
                    key=lambda c: c[0])
    t0, null, spans, n, c = [], [], [], 0, 0
    for gap in range(len(bg) + 1):
        while c < len(copies) and copies[c][0] == gap:
            wt, wn, s, e = docs[copies[c][1]]
            spans.append((copies[c][1], n, n + s, n + e))
            t0 += [wt]
            null += [wn]
            n, c = n + len(wt), c + 1
        if gap < len(bg):
            t0 += [bg[gap]]
            null += [bg[gap]]
            n += len(bg[gap])
    t0, null = np.concatenate(t0), np.concatenate(null)

    # Integrity assertions
    keep = np.ones(len(t0), bool)
    for wid, d, s, e in spans:
        assert np.array_equal(t0[d:e], docs[wid][0][:docs[wid][3]])
        keep[s:e] = False
    assert np.array_equal(t0[keep], null), "T0 minus canary tokens differs from null"
    assert not null_hits([r for a in bg_rows for r in a] + [w["null_text"] for w in works], works)

    pos = {}
    for wid, d, s, e in spans:
        cell = pos.setdefault(f"{works[wid]['format']}/r={works[wid]['repetition']}", [0, 0, 0, 0])
        flags = (1, s // BLOCK != (e - 1) // BLOCK, d // BLOCK == (e - 1) // BLOCK, e > len(t0) // BLOCK * BLOCK)
        cell[:] = [x + y for x, y in zip(cell, flags)]
    write_json(path("corpora", "checks.json"), dict(
        tokens=dict(background=int(sum(map(len, bg))), T0=int(len(t0)), null=int(len(null)), T0_blocks=int(len(t0) // BLOCK)),
        copies_split_wholework_droppedtail=pos))
    write_json(path("corpora", "copies.json"), spans)
    np.save(path("corpora", "null_stream.npy"), null)
    np.save(path("corpora", "T0_stream.npy"), t0)
