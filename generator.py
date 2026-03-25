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

    def _ensure_nltk():
        resources = [
            "taggers/averaged_perceptron_tagger_eng",
            "taggers/averaged_perceptron_tagger",
            "tokenizers/punkt",
            "tokenizers/punkt_tab",
        ]
        for r in resources:
            try:
                nltk.data.find(r)
            except LookupError:
                name = r.split("/")[-1]
                print("[NLTK] downloading", name)
                nltk.download(name, quiet=True)

    _ensure_nltk()
    _POS_AVAILABLE = True

except ImportError:
    _POS_AVAILABLE = False

# ─────────────────────────────
# POS grammar constants
# ─────────────────────────────
_NOUNS = frozenset({"NN", "NNS", "NNP", "NNPS"})
_VERBS = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"})
_ADJECTIVES = frozenset({"JJ", "JJR", "JJS"})

# POS patterns that signal a dangling fragment opener
_BAD_LEAD_POS = frozenset({"IN", "CC", "RB"})

# Known nouns that NLTK sometimes mis-tags as verbs
_NOUN_LEXICON = {n.lower() for n in nouns}
_NOUN_LEXICON.add("suite")

# ─────────────────────────────
# Pre-compiled regex patterns
# ─────────────────────────────
_RE_MULTI_SPACE = re.compile(r"\s{2,}")
_RE_REPEATED_WORD = re.compile(r"\b(\w+) \1\b", re.IGNORECASE)
_RE_TRAILING_PUNC = re.compile(r"[.!?]$")

# ─────────────────────────────
# Paths
# ─────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.json")
HASH_PATH = os.path.join(BASE_DIR, "model.hash")
SEED_PATH = os.path.join(BASE_DIR, "seed_text.py")

_STARTWORDS = frozenset("a the to until".split())

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
        adj = random.choice(adjectives)
        noun = random.choice(nouns)
        verb = random.choice(verbs)
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
    seed2_text = make_seed_sentences(2000)
    model2 = markovify.NewlineText(clean(seed2_text), state_size=3)
    seed3_text = make_short_seed_sentences(4000)
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

_model_lock = threading.Lock()

# ─────────────────────────────
# Verbatim corpus lines
# ─────────────────────────────
_CORPUS_LINES: set = set()
_CORPUS_LINES_LIST: list = []

if os.path.exists(SEED_PATH):
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        for _line in f:
            _stripped = _line.strip().rstrip(".!?").lower()
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
# Candidate filter using RapidFuzz
# ─────────────────────────────
def rapidfuzz_filter(candidates: list[str], threshold: int = 70) -> list[str]:
    """
    Reject candidates whose similarity to any corpus line meets or exceeds
    `threshold`. Uses score_cutoff so rapidfuzz can exit early per candidate.
    """
    if not _CORPUS_LINES_LIST:
        return candidates

    filtered = []
    for s in candidates:
        normalized = s.strip().rstrip(".!?").lower()
        if normalized in _CORPUS_LINES:
            continue
        match = process.extractOne(
            normalized,
            _CORPUS_LINES_LIST,
            scorer=fuzz.ratio,
            score_cutoff=threshold,
        )
        if match is None:
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

    first_word = words[0].lower().rstrip(".,;:")
    if first_word in _FRAGMENT_OPENERS:
        return False, None

    normalized = s.strip().rstrip(".!?").lower()
    if normalized in _CORPUS_LINES:
        return False, None

    pos_ok, _ = _pos_validate(s)
    if not pos_ok:
        return False, None

    s_fixed = s[0].upper() + s[1:]
    if not _RE_TRAILING_PUNC.search(s_fixed.rstrip()):
        s_fixed += "."

    return True, s_fixed

# ─────────────────────────────
# Scoring function
# ─────────────────────────────
def _content_word_set(text: str) -> set[str]:
    words = []
    for w in text.lower().split():
        w = w.strip(string.punctuation)
        if w and w not in _STOPWORDS:
            words.append(w)
    return set(words)

def _score(
    s: str,
    seed_words: set,
    target_noun: str,
    reference_words: set | None = None,
) -> float:
    s_lower = s.lower()
    wds = s_lower.split()
    n = len(wds)

    seed_score = len(set(wds) & seed_words) * 2.0
    content_words = [w.strip(string.punctuation) for w in wds if w.strip(string.punctuation) not in _STOPWORDS]
    richness = len(set(content_words)) / max(len(content_words), 1)
    noun_bonus = 3.0 if target_noun and target_noun in s_lower else -2.0
    length_penalty = 1.0 / (1.0 + abs(n - 10) * 0.4)

    generic_openers = {"a ", "the ", "it ", "this ", "that "}
    opener_penalty = -0.2 if any(s_lower.startswith(o) for o in generic_openers) else 0.0
    if s_lower.startswith("build a") or s_lower.startswith("create a"):
        opener_penalty -= 0.8

    _, grammar_bonus = _pos_validate(s)

    reference_bonus = 0.0
    if reference_words:
        overlap = set(content_words) & reference_words
        reference_bonus = len(overlap) * 1.8

    return (
        seed_score
        + richness * 2.0
        + noun_bonus
        + length_penalty
        + opener_penalty
        + grammar_bonus
        + reference_bonus
    )

# ─────────────────────────────
# Thread-safe single-attempt generators
# ─────────────────────────────
def _try_seeded(seed_word: str, tries: int = 8, reference_words: set | None = None) -> list[str]:
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
# Main generator
# ─────────────────────────────
def generate_best(
    seed: str,
    noun: str = "",
    adj: str = "",
    context_text: str = "",
    n_candidates: int = 60,
    workers: int = 4,
) -> str:
    candidates: list[str] = []
    seed_words = _content_word_set(seed)
    context_words = _content_word_set(context_text)
    words_priority = [w for w in (noun + " " + adj).lower().split() if len(w) > 3]
    target_noun = noun.lower()

    # Phase 1: seeded candidates in parallel
    seeded_words = words_priority[:5] or list(seed_words)[:5]
    with ThreadPoolExecutor(max_workers=min(workers, len(seeded_words) or 1)) as ex:
        futures = {
            ex.submit(_try_seeded, w, n_candidates // max(len(seeded_words), 1)): w
            for w in seeded_words
        }
        for fut in as_completed(futures):
            candidates.extend(fut.result())

    # Phase 2: random fill in parallel
    needed = max(30 - len(candidates), 0)
    random_budget = needed + 10
    random_batch = max(random_budget // workers, 4)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_try_random, seed_words, random_batch) for _ in range(workers)]
        for fut in as_completed(futures):
            candidates.extend(fut.result())

    # Phase 3: rapidfuzz de-duplication
    candidates = rapidfuzz_filter(candidates, threshold=70)

    # Phase 4: score and select
    if candidates:
        best = max(candidates, key=lambda s: _score(s, seed_words, target_noun, context_words))
        best = _RE_MULTI_SPACE.sub(" ", best)
        best = _RE_REPEATED_WORD.sub(r"\1", best)
        best = best.strip()
        return best if _RE_TRAILING_PUNC.search(best) else best + "."

    # Fallback
    seed_parts = seed.split(None, 3)
    if len(seed_parts) == 4:
        _, noun_f, verb_f, twist_f = seed_parts
        return f"A {noun_f} that {verb_f} {twist_f}."
    return f"A tool that {seed.strip()}."

# ─────────────────────────────
# Replace repeated start with pronoun
# ─────────────────────────────
def replace_till_verb(sent1: str, sent2: str, pronoun: str = "It") -> str:
    """
    Replace the opening noun-phrase-like part of sent2 with a pronoun.
    Keeps POS tagging, but avoids cutting on gerunds/participles used as modifiers.
    Also uses a verb lexicon to catch mis-tagged verbs.
    """
    print("\n--- CALL ---")
    print("sent1:", sent1)
    print("sent2:", sent2)

    words2 = sent2.split()
    words1 = sent1.split()
    print("words2:", words2)

    if not words2:
        print("empty sent2")
        return sent2

    def _norm(w: str) -> str:
        return w.strip(string.punctuation).lower()

    # Shared prefix length
    prefix_len = 0
    for a, b in zip(words1, words2):
        if _norm(a) != _norm(b):
            break
        prefix_len += 1
    print("shared prefix len:", prefix_len)

    tags = []
    if _POS_AVAILABLE:
        from nltk import pos_tag
        try:
            tags = pos_tag(words2)
        except Exception as e:
            print("POS error:", e)
            tags = []

    if tags:
        print("TAGS:")
        for w, t in tags:
            print(f"{w:15} {t}")

    # POS sets
    NOUNS = {"NN", "NNS", "NNP", "NNPS"}
    ADJ = {"JJ", "JJR", "JJS"}
    DET = {"DT", "PRP$", "POS"}
    CONNECT = {"IN", "WDT", "WP", "WRB", "TO", "CC"}
    PRON = {"PRP"}
    VERBS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"}
    ADV = {"RB", "RBR", "RBS"}

    CONNECT_WORDS = {
        "that", "which", "who", "whom", "where", "when", "why", "how",
        "because", "since", "although", "unless", "until", "while", "if",
        "as", "before", "after"
    }

    # Override lexicons (replace with your actual nouns and verbs lists)
    _NOUN_LEXICON = {n.lower() for n in nouns}
    _NOUN_LEXICON.add("suite")  # special cases
    _VERB_LEXICON = {v.lower() for v in verbs}

    def is_nounish_token(word: str, tag: str) -> bool:
        w = _norm(word)
        if tag in NOUNS or tag in ADJ or tag in DET or tag in ADV or tag in PRON:
            return True
        if w in _NOUN_LEXICON:
            return True
        return False

    cut_index = 0
    if tags:
        print("\nScanning tags...")
        seen_nounish = False

        for i, (word, tag) in enumerate(tags):
            word_norm = _norm(word)
            next_tag = tags[i + 1][1] if i + 1 < len(tags) else ""
            next_word = _norm(tags[i + 1][0]) if i + 1 < len(tags) else ""

            print(f"i={i}  word={word}  tag={tag}")

            # Normal noun-phrase material
            if is_nounish_token(word, tag):
                seen_nounish = True
                print("allowed noun-phrase tag")
                continue

            # Connectors
            if word_norm in CONNECT_WORDS or tag in CONNECT:
                print("connector found")
                if seen_nounish or i >= 2:
                    cut_index = i + 1
                    break
                continue

            # Participles used as adjectives
            if tag in {"VBG", "VBN"} and (
                next_tag in NOUNS or next_tag in ADJ or next_word in _NOUN_LEXICON
            ):
                print("participial modifier before noun → keep scanning")
                seen_nounish = True
                continue

            # Real verb boundary
            if tag in VERBS or word_norm in _VERB_LEXICON:
                if word_norm in _NOUN_LEXICON:
                    print("verb-tagged but known noun → keep scanning")
                    seen_nounish = True
                    continue

                if seen_nounish or prefix_len > 0 or i >= 2:
                    print("FOUND VERB at", i)
                    cut_index = i
                    break

                print("leading verb-like token ignored")
                continue

            print("NOT ALLOWED → stop replacement")
            cut_index = 0
            break

    if cut_index == 0 and prefix_len >= 2 and prefix_len < len(words2):
        cut_index = prefix_len

    print("cut_index =", cut_index)

    if cut_index > 0 and cut_index < len(words2):
        remainder = words2[cut_index:]
        print("remainder:", remainder)
        replaced = pronoun.capitalize() + " " + " ".join(remainder)
        print("RESULT:", replaced)
        return replaced

    print("No replacement")
    return sent2

# ─────────────────────────────
# Public helpers
# ─────────────────────────────
def generate_coding_idea():
    adj = random.choice(adjectives) if random.random() < 0.9 else ""
    noun = random.choice(nouns)
    verb = random.choice(verbs)
    twist = random.choice(twists) if random.random() < 0.9 else ""
    return adj, noun, verb, twist

def get_description(adj, noun, verb, twist):
    seed = f"{adj} {noun} {verb} {twist}".strip()
    sent1 = generate_best(seed, noun=noun, adj=adj)

    # Sentence 2 should be rewarded for overlapping both the idea seed and sent1
    sent2 = ""
    for _ in range(6):
        candidate = generate_best(seed, noun=noun, adj=adj, context_text=sent1)
        if fuzz.ratio(sent1.lower(), candidate.lower()) < 88:
            sent2 = candidate
            break
        sent2 = candidate

    sent2 = replace_till_verb(sent1, sent2, pronoun="It")

    return sent1 + " " + sent2

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