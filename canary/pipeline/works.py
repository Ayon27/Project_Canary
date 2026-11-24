"""Stage works: select canary and held-out works, insert canaries, build null texts and judge prefixes."""
import random

from ..data.wikitext import passage, train_articles
from ..utils import encode, out, write_json, write_jsonl
from .canaries import canary_doc, candidate_tokens, split_sentences


def run(c):
    articles, E, B, W = train_articles(), c["eos"] + "\n", c["block"], c["works_per_group"]
    title = lambda k: articles[k][0].strip(" =\n")

    # Eligible: enough sentences, and the canaried work fits one block.
    sents = [split_sentences(passage(a)) for a in articles]
    ok = [k for k, s in enumerate(sents) if len(s) >= c["min_sentences"]]
    lens = [len(ids) for i in range(0, len(ok), 1000)
            for ids in encode(c, [E + " ".join(sents[k]) + "\n" for k in ok[i:i + 1000]])]
    eligible = [k for k, n in zip(ok, lens) if n + c["length_margin"] <= B]

    # Skip works that share a long row with another article.
    owner, shared = {}, set()
    for k, a in enumerate(articles):
        for r in {r.strip() for r in a}:
            if owner.setdefault(r, k) != k:
                shared.add(r)
    random.Random(c["selection_seed"]).shuffle(eligible)
    chosen, rejected, need = [], [], len(c["groups"]) * W + c["n_heldout"]
    for k in eligible:
        if len(chosen) == need:
            break
        rows = list({r.strip() for r in articles[k]} & shared)
        (rejected if rows and max(map(len, encode(c, rows))) >= c["min_leak_tokens"] else chosen).append(k)
    assert len(chosen) == need, f"only {len(chosen)} usable works"
    canary, heldout = chosen[:-c["n_heldout"]], chosen[-c["n_heldout"]:]
    random.Random(c["group_seed"]).shuffle(canary)

    records = []
    for g, (fmt, r) in enumerate(c["groups"]):
        for i, k in enumerate(canary[g * W:(g + 1) * W]):
            text, cands, idx = canary_doc(passage(articles[k]), fmt, g, i, c["K"])
            line = "\n" + cands[idx] + "\n"
            assert text.count(line) == 1 and text.count("\n") == 2
            null, prefix = text.replace(line, "\n", 1), E + text[:text.index(line) + 1]
            _, tails = candidate_tokens(c, prefix, cands)
            assert len(encode(c, E + null + "\n")) + max(map(len, tails)) <= B, k
            records.append(dict(work_id=g * W + i, group=g, format=fmt, repetition=r, seed=g, article_index=k,
                                title=title(k), original=passage(articles[k]), t0_text=text, null_text=null,
                                judge_prefix=prefix, released_idx=idx, released=cands[idx], candidates=cands))

    write_json(out(c, "config.json"), dict(
        selection_seed=c["selection_seed"], group_seed=c["group_seed"], canary_seeds=list(range(len(c["groups"]))),
        insertion_seed=c["insertion_seed"], train_seed=c["train_seed"], groups=c["groups"]))
    write_json(out(c, "works", "selection_log.json"), dict(articles=len(articles), eligible=len(eligible),
                                                         rejected_for_leakage=rejected))
    write_jsonl(out(c, "works", "heldout_works.jsonl"),
                [dict(article_index=k, title=title(k), original=passage(articles[k])) for k in heldout])
    write_jsonl(out(c, "works", "works_manifest.jsonl"), records)  # written last: marks the stage as done
