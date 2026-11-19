"""Random-string canaries in a randomized K-of-1 design (Zhang et al. 2024, Lemma 1).

For each work, K candidates are drawn from the same procedure and one is released uniformly at random.
Three formats: rand_alnum, rand_words, carrier. The released canary is inserted on its own line in the
middle third of the work. All K candidates and the released index are stored, so the judge can score every
candidate with the trained model and compute the rank p-value.

rank_pvalue treats a HIGHER score as stronger evidence of training. With model loss, pass the negative loss
(lower loss = the model finds the candidate more familiar).

Extracted from Ruyu Zhou's Pilot 0 code (canaries.py, utils.py, watermark.py). Text-level survival
functions and fictitious facts are omitted.
"""
import random
import re
import string

DEFAULT_CFG = {
    "K": 100,
    "formats": ["rand_alnum", "rand_words", "carrier"],
    "own_line": True,
    "alnum_len": 20,
    "n_words": 8,
    "carrier_template": "Reference code: {code}.",
    "carrier_code_len": 12,
}


ALNUM = string.ascii_letters + string.digits
COMMON_WORDS = """time year people way day man thing woman life child world school state family student group
country problem hand part place case week company system program question work government number night point
home water room mother area money story fact month lot right study book eye job word business issue side kind
head house service friend father power hour game line end member law car city community name president team
minute idea kid body information back parent face others level office door health person art war history party
result change morning reason research girl guy moment air teacher force education foot boy age policy music
market sense nation plan college interest death experience effect class control care field development role
effort rate heart drug show leader light voice wife police mind price report decision son view relationship
town road arm difference value building action model season society tax director position player record paper
space ground form event official matter center couple site project activity star table need court oil
situation cost industry figure street image phone data picture practice piece land product doctor wall patient
worker news test movie north love support technology step baby computer type attention film tree source organ
garden river window lantern velvet orbit marble copper meadow harbor pencil ladder candle thunder silver""".split()

def _alnum(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(ALNUM) for _ in range(n))

_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])')

def split_sentences(text: str) -> list[str]:
    """Newlines are hard boundaries (canaries are inserted on their own line); then split at sentence ends."""
    out = []
    for line in text.strip().split("\n"):
        out += [s.strip() for s in _SENT_RE.split(line.strip()) if s.strip()]
    return out

def make_candidates(fmt: str, rng: random.Random, K: int, ccfg: dict) -> list[dict]:
    cands = []
    for _ in range(K):
        if fmt == "rand_alnum":
            key = _alnum(rng, ccfg["alnum_len"])
            sent = key + "."
        elif fmt == "rand_words":
            key = " ".join(rng.sample(COMMON_WORDS, ccfg["n_words"]))
            sent = key[0].upper() + key[1:] + "."
        elif fmt == "carrier":
            key = _alnum(rng, ccfg["carrier_code_len"])
            sent = ccfg["carrier_template"].format(code=key)
        else:
            raise ValueError(fmt)
        cands.append({"key": key, "sentence": sent, "attrs": None})
    return cands

def insert_sentence(text: str, sentence: str, rng: random.Random, own_line: bool = False) -> str:
    """Insert in the middle third. own_line=True puts it on a separate line (as a code / hidden string would be)."""
    sents = split_sentences(text)
    n = len(sents)
    lo = max(1, n // 3)
    hi = max(lo, (2 * n) // 3)
    pos = rng.randint(lo, hi)
    if own_line:
        return " ".join(sents[:pos]) + "\n" + sentence + "\n" + " ".join(sents[pos:])
    return " ".join(sents[:pos] + [sentence] + sents[pos:])

def build_canary_docs(passages: list[str], ccfg: dict, seed: int) -> list[dict]:
    rows = []
    for fmt in ccfg["formats"]:
        for i, p in enumerate(passages):
            rng = random.Random(f"{seed}-canary-{fmt}-{i}")
            cands = make_candidates(fmt, rng, ccfg["K"], ccfg)
            idx = rng.randrange(len(cands))
            rows.append(
                dict(
                    uid=f"canary/{fmt}/{i}",
                    kind="canary",
                    fmt=fmt,
                    doc_id=i,
                    original=p,  # canary-free work (reference for content retention)
                    text=insert_sentence(p, cands[idx]["sentence"], rng,
                                         own_line=ccfg.get("own_line", True)),
                    released_idx=idx,
                    canary_key=cands[idx]["key"],
                    attrs=cands[idx]["attrs"],
                    candidates=[c["sentence"] for c in cands],
                )
            )
    return rows

def rank_pvalue(values: list[float], released_idx: int) -> float:
    """Conservative K-of-1 rank p-value: fraction of candidates scoring >= the released one (Lemma 1)."""
    v = values[released_idx]
    if v != v:  # nan
        return 1.0
    return sum(1 for x in values if x >= v) / len(values)

if __name__ == "__main__":
    # Smoke test: build canary docs for two toy passages and check the design.
    passages = [
        "The river rises in the hills. It flows east for forty miles. Several towns lie on its banks. "
        "A bridge was built in 1820. The bridge still stands today. Fishing remains common in summer.",
        "The museum opened in 1901. Its first director was a botanist. The collection grew quickly. "
        "A new wing was added in 1950. The museum now holds over a million items. Entry is free on Sundays.",
    ]
    docs = build_canary_docs(passages, DEFAULT_CFG, seed=0)
    for d in docs:
        released = d["candidates"][d["released_idx"]]
        assert released in d["text"] and len(d["candidates"]) == DEFAULT_CFG["K"]
    print(docs[0]["text"], "\n")
    print(f"{len(docs)} docs built; released candidate found in every text.")

    # Rank p-value sanity check: with scores independent of the released index, p is roughly uniform.
    rng = random.Random(1)
    ps = [rank_pvalue([rng.random() for _ in range(100)], rng.randrange(100)) for _ in range(5000)]
    print("fraction of p <= 0.01 under the null:", sum(p <= 0.01 for p in ps) / len(ps), "(should be about 0.01)")
