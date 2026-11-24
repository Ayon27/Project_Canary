"""Stage corpora: T0 and null token streams.

Background articles keep their order. Each canary work gets `repetition` copies at random article gaps;
null has the same copies with the canary removed.
"""
import itertools
import random

import numpy as np

from ..data.wikitext import train_articles
from ..utils import encode, out, read_jsonl, write_json
from .canaries import candidate_tokens


def encode_background(c, articles):
    res = []
    for i in range(0, len(articles), 256):
        chunk = articles[i:i + 256]
        ids = encode(c, [c["eos"] + "\n" + r if j == 0 else r for a in chunk for j, r in enumerate(a)])
        ends = np.cumsum([len(a) for a in chunk])
        res += [np.fromiter(itertools.chain(*ids[e - len(a):e]), c["dtype"]) for a, e in zip(chunk, ends)]
    return res


def encode_work(c, w):
    """T0 and null token documents, and the canary span [s, e) in T0."""
    E = c["eos"] + "\n"
    t0, null = encode(c, [E + w["t0_text"] + "\n", E + w["null_text"] + "\n"])
    p, (tail,) = candidate_tokens(c, w["judge_prefix"], [w["released"]])
    s, e = len(p), len(p) + len(tail)
    assert t0[:e] == p + tail and t0[:s] + t0[e:] == null, w["work_id"]
    return np.array(t0, c["dtype"]), np.array(null, c["dtype"]), s, e


def null_hits(texts, works):
    """Texts that contain any candidate (its code, or its 8 words for rand_words)."""
    keys = {tuple(c[:-1].lower().split()) if w["format"] == "rand_words" else (c[:-1].split()[-1].lower(),)
            for w in works for c in w["candidates"]}
    first, hits = {k[0] for k in keys}, []
    for t in texts:
        ws = t.lower().split()
        hits += [t for i, x in enumerate(ws) if x in first and ({tuple(ws[i:i + 1]), tuple(ws[i:i + 8])} & keys)]
    return hits


def run(c):
    works, held = read_jsonl(out(c, "works", "works_manifest.jsonl")), read_jsonl(out(c, "works", "heldout_works.jsonl"))
    excluded = {w["article_index"] for w in works + held}
    bg_rows = [a for k, a in enumerate(train_articles()) if k not in excluded]
    bg, docs, B = encode_background(c, bg_rows), [encode_work(c, w) for w in works], c["block"]

    rng = random.Random(c["insertion_seed"])
    copies = sorted(((rng.randint(0, len(bg)), w["work_id"]) for w in works for _ in range(w["repetition"])),
                    key=lambda x: x[0])
    t0, null, spans, n, i = [], [], [], 0, 0  # spans: (work_id, doc start, canary start, canary end)
    for gap in range(len(bg) + 1):
        while i < len(copies) and copies[i][0] == gap:
            wt, wn, s, e = docs[copies[i][1]]
            spans.append((copies[i][1], n, n + s, n + e))
            t0.append(wt)
            null.append(wn)
            n, i = n + len(wt), i + 1
        if gap < len(bg):
            t0.append(bg[gap])
            null.append(bg[gap])
            n += len(bg[gap])
    t0, null = np.concatenate(t0), np.concatenate(null)

    keep = np.ones(len(t0), bool)
    for wid, d, s, e in spans:
        assert np.array_equal(t0[d:e], docs[wid][0][:docs[wid][3]])
        keep[s:e] = False
    assert np.array_equal(t0[keep], null), "T0 minus the canary tokens differs from null"
    assert not null_hits([r for a in bg_rows for r in a] + [w["null_text"] for w in works], works)

    # Per group: [copies, canary split across blocks, work within one block, in the dropped tail].
    pos = {}
    for wid, d, s, e in spans:
        cell = pos.setdefault(f"{works[wid]['format']}/r={works[wid]['repetition']}", [0, 0, 0, 0])
        flags = (1, s // B != (e - 1) // B, d // B == (e - 1) // B, e > len(t0) // B * B)
        cell[:] = [x + y for x, y in zip(cell, flags)]
    write_json(out(c, "corpora", "checks.json"), dict(
        tokens=dict(background=int(sum(map(len, bg))), T0=len(t0), null=len(null), T0_blocks=len(t0) // B),
        copies_split_wholework_droppedtail=pos))
    write_json(out(c, "corpora", "copies.json"), spans)
    np.save(out(c, "corpora", "null_stream.npy"), null)
    np.save(out(c, "corpora", "T0_stream.npy"), t0)  # written last: marks the stage as done
