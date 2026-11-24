"""Random-string canaries, K-of-1 design (Zhang et al. 2024, Lemma 1), from Ruyu Zhou's Pilot 0 code."""
import random
import re
import string

from ..utils import encode, tokenizer

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
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])')


def split_sentences(text):
    out = []
    for line in text.strip().split("\n"):
        out += [s.strip() for s in _SENT_RE.split(line.strip()) if s.strip()]
    return out


def candidate(fmt, rng):
    alnum = lambda n: "".join(rng.choice(ALNUM) for _ in range(n))
    if fmt == "rand_alnum":
        return alnum(20) + "."
    if fmt == "rand_words":
        s = " ".join(rng.sample(COMMON_WORDS, 8))
        return s[0].upper() + s[1:] + "."
    if fmt == "carrier":
        return f"Reference code: {alnum(12)}."
    raise ValueError(fmt)


def canary_doc(passage, fmt, seed, i, K):
    """K candidates; a random one is inserted on its own line in the middle third. Returns (text, cands, idx)."""
    rng = random.Random(f"{seed}-canary-{fmt}-{i}")
    cands = [candidate(fmt, rng) for _ in range(K)]
    idx = rng.randrange(K)
    sents = split_sentences(passage)
    lo = max(1, len(sents) // 3)
    pos = rng.randint(lo, max(lo, 2 * len(sents) // 3))
    return " ".join(sents[:pos]) + "\n" + cands[idx] + "\n" + " ".join(sents[pos:]), cands, idx


def rank_pvalue(values, released_idx):
    """Fraction of candidates scoring >= the released one (higher = more familiar)."""
    v = values[released_idx]
    return 1.0 if v != v else sum(x >= v for x in values) / len(values)


def candidate_tokens(c, prefix, candidates):
    """Ids of the prefix, and of each candidate in prefix + candidate + "\\n". Checks the prefix ids are shared and
    the tail decodes to the candidate (ignoring spaces unless exact_decode)."""
    p, dec = encode(c, prefix), tokenizer(c["tokenizer"]).decode
    norm = lambda s: s.replace(" ", "").strip()
    tails = []
    for cand in candidates:
        ids = encode(c, prefix + cand + "\n")
        tail = ids[len(p):]
        ok = (dec(tail) == cand + "\n" and dec(ids[-1:]) == ".\n") if c["exact_decode"] else \
            (norm(dec(tail)) == norm(cand) and dec(ids[-1:]).strip() == ".")
        assert ids[:len(p)] == p and ok, cand
        tails.append(tail)
    return p, tails
