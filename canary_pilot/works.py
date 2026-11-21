"""Stage works: pick 900 works (450 canary + 450 held out), 9 groups of 50, canaries, null texts, judge prefixes."""
import random

from canaries import DEFAULT_CFG, build_canary_docs, split_sentences

from .common import (BLOCK, EOS, GROUP_SEED, GROUPS, INSERTION_SEED, LENGTH_MARGIN, MIN_LEAK_TOKENS, MIN_SENTENCES,
                     N_HELDOUT, SELECTION_SEED, TRAIN_SEED, WORKS_PER_GROUP, candidate_tokens, encode, passage, path,
                     train_articles, write_json, write_jsonl)


def run():
    articles = train_articles()

    # Eligible: >= 10 sentences, and EOS\n + canaried text + \n fits one block with room for the longest candidate.
    sents = [split_sentences(passage(a)) for a in articles]
    ok = [k for k, s in enumerate(sents) if len(s) >= MIN_SENTENCES]
    lens = [len(ids) for i in range(0, len(ok), 1000)
            for ids in encode([EOS + "\n" + " ".join(sents[k]) + "\n" for k in ok[i:i + 1000]])]
    eligible = [k for k, n in zip(ok, lens) if n + LENGTH_MARGIN <= BLOCK]

    # Leakage: skip a work if one of its rows of >= 20 tokens also occurs in another article.
    owner, shared = {}, set()
    for k, a in enumerate(articles):
        for r in {r.strip() for r in a}:
            if owner.setdefault(r, k) != k:
                shared.add(r)
    random.Random(SELECTION_SEED).shuffle(eligible)
    chosen, rejected, need = [], [], len(GROUPS) * WORKS_PER_GROUP + N_HELDOUT
    for k in eligible:
        if len(chosen) == need:
            break
        rows = list({r.strip() for r in articles[k]} & shared)
        (rejected if rows and max(map(len, encode(rows))) >= MIN_LEAK_TOKENS else chosen).append(k)
    assert len(chosen) == need, f"only {len(chosen)} usable works"
    canary, heldout = chosen[:-N_HELDOUT], chosen[-N_HELDOUT:]
    random.Random(GROUP_SEED).shuffle(canary)

    records = []
    for g, (fmt, r) in enumerate(GROUPS):
        ks = canary[g * WORKS_PER_GROUP:(g + 1) * WORKS_PER_GROUP]
        docs = build_canary_docs([passage(articles[k]) for k in ks], {**DEFAULT_CFG, "formats": [fmt]}, seed=g)
        for i, (k, d) in enumerate(zip(ks, docs)):
            text, released = d["text"], d["candidates"][d["released_idx"]]
            line = "\n" + released + "\n"
            assert text.count(line) == 1 and text.count("\n") == 2  # insert_sentence joins sentences with spaces
            null = text.replace(line, "\n", 1)
            prefix = EOS + "\n" + text[:text.index(line) + 1]
            _, tails = candidate_tokens(prefix, d["candidates"])
            assert len(encode(EOS + "\n" + null + "\n")) + max(map(len, tails)) <= BLOCK, k
            records.append(dict(work_id=g * WORKS_PER_GROUP + i, group=g, format=fmt, repetition=r, seed=g,
                                article_index=k, title=articles[k][0].strip(" =\n"), original=d["original"],
                                t0_text=text, null_text=null, judge_prefix=prefix, released_idx=d["released_idx"],
                                released=released, candidates=d["candidates"]))

    write_json(path("config.json"), dict(selection_seed=SELECTION_SEED, group_seed=GROUP_SEED, canary_seeds=list(
        range(len(GROUPS))), insertion_seed=INSERTION_SEED, train_seed=TRAIN_SEED, groups=GROUPS))
    write_json(path("works", "selection_log.json"), dict(articles=len(articles), eligible=len(eligible),
                                                         rejected_for_leakage=rejected))
    write_jsonl(path("works", "heldout_works.jsonl"), [dict(article_index=k, title=articles[k][0].strip(" =\n"),
                                                            original=passage(articles[k])) for k in heldout])
    write_jsonl(path("works", "works_manifest.jsonl"), records)  # written last: marks the stage as done
