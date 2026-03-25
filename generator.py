import markovify
import random
import string
import os
import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from rapidfuzz import fuzz, process
from word_lists import adjectives, nouns, verbs, twists

# ─────────────────────────────
# Optional POS support (nltk)
# ─────────────────────────────
try:
    import nltk
    from nltk import pos_tag, word_tokenize

    for _res, _kind in [
        ("averaged_perceptron_tagger", "taggers"),
        ("punkt_tab", "tokenizers"),
    ]:
        try:
            nltk.data.find(f"{_kind}/{_res}")
        except LookupError:
            nltk.download(_res, quiet=True)

    _POS_AVAILABLE = True
except ImportError:
    _POS_AVAILABLE = False

# ─────────────────────────────
# POS grammar constants
# ─────────────────────────────
_NOUNS      = frozenset({"NN", "NNS", "NNP", "NNPS"})
_VERBS      = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"})
_ADJECTIVES = frozenset({"JJ", "JJR", "JJS"})

# POS patterns that signal a dangling fragment opener
_BAD_LEAD_POS = frozenset({"IN", "CC", "RB"})  # subordinating conj, coord conj, adverb-opener

# ─────────────────────────────
# Pre-compiled regex patterns
# ─────────────────────────────
_RE_MULTI_SPACE   = re.compile(r"\s{2,}")
_RE_REPEATED_WORD = re.compile(r"\b(\w+) \1\b", re.IGNORECASE)
_RE_TRAILING_PUNC = re.compile(r"[.!?]$")

# ─────────────────────────────
# Paths
# ─────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.json")
HASH_PATH  = os.path.join(BASE_DIR, "model.hash")
SEED_PATH  = os.path.join(BASE_DIR, "seed_text.py")

_STARTWORDS = frozenset("a the to until ")

_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with that this is are was were "
    "it its they them their be been being have has had do does did will would "
    "could should may might shall not so as if by from up out just also then "
    "than when what which who how all any some such no each".split()
)

_FRAGMENT_OPENERS = frozenset(
    "retrospectively additionally furthermore however moreover therefore consequently "
    "subsequently alternatively accordingly nevertheless nonetheless whereas despite "
    "although because since unless until whenever wherever whoever whatever "
    "combining using helping allowing enabling reducing improving supporting "
    "providing giving offering making letting".split()
)

_BAD_ENDERS = frozenset({
    "so", "because", "that", "which", "who", "whom", "where", "when",
    "although", "unless", "while", "if", "as", "after", "before", "until"
})

# ─────────────────────────────
# Clean text
# ─────────────────────────────
def clean(text: str) -> str:
    allowed = set(string.ascii_letters + string.digits + " .!?\n,;:-'")
    return "".join(c if c in allowed else " " for c in text).lower()


# ─────────────────────────────
# Corpus hash / rebuild check
# ─────────────────────────────
def _corpus_hash() -> str:
    h = hashlib.md5()
    for path in [SEED_PATH, os.path.join(BASE_DIR, "word_lists.py")]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def _model_is_fresh() -> bool:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(HASH_PATH):
        return False
    with open(HASH_PATH) as f:
        return f.read().strip() == _corpus_hash()


# ─────────────────────────────
# Generate synthetic sentences
# ─────────────────────────────
def make_seed_sentences(n_sentences: int = 500) -> str:
    sentences = []
    for _ in range(n_sentences):
        adj   = random.choice(adjectives)
        noun  = random.choice(nouns)
        verb  = random.choice(verbs)
        twist = random.choice(twists)
        sentences.append(f"A {adj} {noun} that {verb} {twist}.")
    return "\n".join(sentences)


def make_short_seed_sentences(n_sentences: int = 500) -> str:
    sentences = []
    for _ in range(n_sentences):
        noun = random.choice(nouns)
        verb = random.choice(verbs)
        sentences.append(f"A {noun} that {verb}.")
    return "\n".join(sentences)


# ─────────────────────────────
# Build model
# ─────────────────────────────
def build_models():
    print("[build_models] Building models …")
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        seed1_text = clean(f.read())

    model1 = markovify.NewlineText(seed1_text, state_size=3)
    seed2_text = make_seed_sentences(500)
    model2 = markovify.NewlineText(clean(seed2_text), state_size=3)
    seed3_text = make_short_seed_sentences(500)
    model3 = markovify.NewlineText(clean(seed3_text), state_size=3)
    combined = markovify.combine([model1, model2, model3], [0.70, 0.10, 0.20])

    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        f.write(combined.to_json())
    with open(HASH_PATH, "w") as f:
        f.write(_corpus_hash())

    print("[build_models] Done")
    return combined


# ─────────────────────────────
# Load or build
# ─────────────────────────────
if _model_is_fresh():
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        model = markovify.NewlineText.from_json(f.read())
else:
    model = build_models()

# Thread-local model access guard (markovify reads are safe, but guard chain state)
_model_lock = threading.Lock()

# ─────────────────────────────
# Verbatim corpus lines (as list for rapidfuzz)
# ─────────────────────────────
_CORPUS_LINES: set = set()
_CORPUS_LINES_LIST: list = []       # rapidfuzz needs a sequence for score_cutoff early-exit

if os.path.exists(SEED_PATH):
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        for _line in f:
            _stripped = _line.strip().rstrip('.!?').lower()
            if len(_stripped) > 20:
                _CORPUS_LINES.add(_stripped)
    _CORPUS_LINES_LIST = list(_CORPUS_LINES)


# ─────────────────────────────
# POS helpers
# ─────────────────────────────
@lru_cache(maxsize=512)
def _pos_tags(sentence: str):
    """Cached POS tagging — sentence is the cache key."""
    if not _POS_AVAILABLE:
        return []
    try:
        return pos_tag(word_tokenize(sentence))
    except Exception:
        return []


def _pos_validate(sentence: str) -> tuple[bool, float]:
    if not _POS_AVAILABLE:
        return True, 0.0

    tags = _pos_tags(sentence)
    if not tags:
        return True, 0.0

    pos_seq = [tag for _, tag in tags]
    pos_set = set(pos_seq)

    # Hard rules: at least one noun AND one verb
    has_noun = bool(pos_set & _NOUNS)
    has_verb = bool(pos_set & _VERBS)
    if not has_noun or not has_verb:
        return False, 0.0

    # Must not open with subordinating/conjunction/adverb
    lead_pos = pos_seq[0] if pos_seq else ""
    if lead_pos in _BAD_LEAD_POS:
        return False, 0.0

    # Reject if ends with a bad ender
    last_word = sentence.split()[-1].lower().rstrip(".,;:")
    if last_word in _BAD_ENDERS:
        return False, 0.0

    # Bonus: reward subject-verb order
    grammar_score = 0.0
    first_noun_idx = next((i for i, t in enumerate(pos_seq) if t in _NOUNS), None)
    first_verb_idx = next((i for i, t in enumerate(pos_seq) if t in _VERBS), None)
    if first_noun_idx is not None and first_verb_idx is not None:
        if first_noun_idx < first_verb_idx:
            grammar_score += 1.5
        else:
            grammar_score -= 0.5

    if pos_set & _ADJECTIVES:
        grammar_score += 0.5

    # Penalize back-to-back same POS
    for a, b in zip(pos_seq, pos_seq[1:]):
        if a == b and a in _NOUNS:
            grammar_score -= 0.4

    return True, grammar_score


def _pos_score_word(word: str) -> str:
    """Return the dominant POS tag for a single word (for seed-word selection)."""
    if not _POS_AVAILABLE:
        return "NN"
    tags = _pos_tags(word)
    return tags[0][1] if tags else "NN"


# ─────────────────────────────
# Candidate filter using RapidFuzz (with score_cutoff for speed)
# ─────────────────────────────
def rapidfuzz_filter(candidates: list[str], threshold: int = 70) -> list[str]:
    """
    Reject candidates whose similarity to any corpus line meets or exceeds
    `threshold`.  Uses score_cutoff so rapidfuzz can exit early per candidate.
    """
    if not _CORPUS_LINES_LIST:
        return candidates

    filtered = []
    for s in candidates:
        normalized = s.strip().rstrip('.!?').lower()
        if normalized in _CORPUS_LINES:
            continue
        # score_cutoff makes extractOne return None when no match reaches threshold
        match = process.extractOne(
            normalized,
            _CORPUS_LINES_LIST,
            scorer=fuzz.ratio,
            score_cutoff=threshold,
        )
        if match is None:       # nothing similar enough → keep
            filtered.append(s)
    return filtered


# ─────────────────────────────
# Enhanced candidate filter (grammar + structure)
# ─────────────────────────────
def good_enhanced(s: str) -> tuple[bool, str | None]:
    if not s:
        return False, None

    words = s.split()
    if not (7 <= len(words) <= 15):
        return False, None
    if len(set(words)) < len(words) * 0.75:
        return False, None

    first_word = words[0].lower().rstrip('.,;:')
    if first_word in _FRAGMENT_OPENERS:
        return False, None

    normalized = s.strip().rstrip('.!?').lower()
    if normalized in _CORPUS_LINES:
        return False, None

    # POS hard structural check (fail fast before string ops)
    pos_ok, _ = _pos_validate(s)
    if not pos_ok:
        return False, None

    s_fixed = s[0].upper() + s[1:]
    if not _RE_TRAILING_PUNC.search(s_fixed.rstrip()):
        s_fixed += '.'

    return True, s_fixed


# ─────────────────────────────
# Scoring function (now includes POS grammar bonus)
# ─────────────────────────────
def _score(s: str, seed_words: set, target_noun: str) -> float:
    s_lower = s.lower()
    wds     = s_lower.split()
    n       = len(wds)

    seed_score    = len(set(wds) & seed_words) * 2.0
    content_words = [w for w in wds if w not in _STOPWORDS]
    richness      = len(set(content_words)) / max(len(content_words), 1)
    noun_bonus    = 3.0 if target_noun and target_noun in s_lower else -2.0
    length_penalty = 1.0 / (1.0 + abs(n - 10) * 0.4)

    generic_openers = {"a ", "the ", "it ", "this ", "that "}
    opener_penalty = -0.2 if any(s_lower.startswith(o) for o in generic_openers) else 0.0
    if s_lower.startswith("build a") or s_lower.startswith("create a"):
        opener_penalty -= 0.8

    # POS grammar bonus (cached, so cheap after first call)
    _, grammar_bonus = _pos_validate(s)

    return (
        seed_score
        + richness * 2.0
        + noun_bonus
        + length_penalty
        + opener_penalty
        + grammar_bonus
    )


# ─────────────────────────────
# Thread-safe single-attempt generators
# ─────────────────────────────
def _try_seeded(seed_word: str, tries: int = 8) -> list[str]:
    """Try to make sentences starting with seed_word; returns valid candidates."""
    results = []
    for _ in range(tries):
        try:
            with _model_lock:
                s = model.make_sentence_with_start(
                    seed_word, strict=False, max_words=15, tries=15, test_output=False
                )
            ok, s_fixed = good_enhanced(s or "")
            if ok:
                results.append(s_fixed)
        except Exception:
            pass
    return results


def _try_random(seed_words: set, tries: int = 8) -> list[str]:
    """Try to make random sentences that share at least one seed word."""
    results = []
    for _ in range(tries):
        try:
            with _model_lock:
                s = model.make_sentence(tries=20, max_words=15, test_output=False, strict=False)
            ok, s_fixed = good_enhanced(s or "")
            if ok and set(s_fixed.lower().split()) & seed_words:
                results.append(s_fixed)
        except Exception:
            pass
    return results


# ─────────────────────────────
# Main generator — parallelised candidate collection
# ─────────────────────────────
def generate_best(
    seed: str,
    noun: str = "",
    adj: str = "",
    n_candidates: int = 25,
    workers: int = 4,
) -> str:
    candidates: list[str] = []
    seed_words    = set(seed.lower().split())
    words_priority = [w for w in (noun + " " + adj).lower().split() if len(w) > 3]
    target_noun   = noun.lower()

    # ── Phase 1: seeded candidates in parallel ──────────────────────────────
    seeded_words = words_priority[:5] or list(seed_words)[:5]
    with ThreadPoolExecutor(max_workers=min(workers, len(seeded_words) or 1)) as ex:
        futures = {ex.submit(_try_seeded, w, n_candidates // max(len(seeded_words), 1)): w
                   for w in seeded_words}
        for fut in as_completed(futures):
            candidates.extend(fut.result())

    # ── Phase 2: random fill in parallel ────────────────────────────────────
    needed        = max(30 - len(candidates), 0)
    random_budget = needed + 10        # slight over-request to compensate for filtering
    random_batch  = max(random_budget // workers, 4)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_try_random, seed_words, random_batch) for _ in range(workers)]
        for fut in as_completed(futures):
            candidates.extend(fut.result())

    # ── Phase 3: rapidfuzz de-duplication ───────────────────────────────────
    candidates = rapidfuzz_filter(candidates, threshold=70)

    # ── Phase 4: score and select ────────────────────────────────────────────
    if candidates:
        best = max(candidates, key=lambda s: _score(s, seed_words, target_noun))
        best = _RE_MULTI_SPACE.sub(" ", best)
        best = _RE_REPEATED_WORD.sub(r"\1", best)
        return best.strip() if _RE_TRAILING_PUNC.search(best[-1]) else best.strip() + "."

    # ── Fallback ─────────────────────────────────────────────────────────────
    seed_parts = seed.split(None, 3)
    if len(seed_parts) == 4:
        _, noun_f, verb_f, twist_f = seed_parts
        return f"A {noun_f} that {verb_f} {twist_f}."
    return f"A tool that {seed.strip()}."


# ─────────────────────────────
# Public helpers
# ─────────────────────────────
def generate_coding_idea():
    adj   = random.choice(adjectives) if random.random() < 0.9 else ""
    noun  = random.choice(nouns)
    verb  = random.choice(verbs)
    twist = random.choice(twists) if random.random() < 0.9 else ""
    return adj, noun, verb, twist


def get_description(adj, noun, verb, twist):
    seed = f"{adj} {noun} {verb} {twist}".strip()
    return generate_best(seed, noun=noun, adj=adj) + " " + generate_best(seed, noun=noun, adj=adj)


# ─────────────────────────────
# Example usage
# ─────────────────────────────
if __name__ == "__main__":
    adj, noun, verb, twist = (
        random.choice(adjectives),
        random.choice(nouns),
        random.choice(verbs),
        random.choice(twists),
    )
    seed = f"{adj} {noun} {verb} {twist}"
    print("Seed   :", seed)
    print("POS OK :", _pos_validate(seed))
    print("Result :", generate_best(seed, noun=noun, adj=adj))