import markovify
import random
import string
import os

from word_lists import adjectives, nouns, verbs, twists

# ─────────────────────────────
# Paths
# ─────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.json")
SEED_PATH = os.path.join(BASE_DIR, "seed_text.py")


# ─────────────────────────────
# Clean text (keep basic punctuation)
# ─────────────────────────────
def clean(text: str) -> str:
    allowed = set(string.ascii_letters + string.digits + " .!?\n,;:-'")
    cleaned = "".join(c if c in allowed else " " for c in text)
    cleaned = cleaned.lower()
    return cleaned


# ─────────────────────────────
# Synthetic sentences (500 for coverage)
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


# ─────────────────────────────
# IMPROVEMENT 1: Higher State Size = 4
# ─────────────────────────────
def build_models():
    print("[build_models] Building models with state_size=4...")

    # Model 1: Original corpus
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        seed1_text = clean(f.read())
    model1 = markovify.Text(seed1_text, state_size=4)  # ↑ from 3

    # Model 2: Synthetic (75% weight)
    seed2_text = make_seed_sentences(500)
    model2 = markovify.Text(clean(seed2_text), state_size=4)

    combined = markovify.combine([model1, model2], [0.25, 0.75])
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        f.write(combined.to_json())
    print("[build_models] State_size=4 model saved")
    return combined


# ─────────────────────────────
# Load model
# ─────────────────────────────
if not os.path.exists(MODEL_PATH):
    model = build_models()
else:
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        model = markovify.Text.from_json(f.read())


# ─────────────────────────────
# IMPROVEMENT 2+3: Enhanced filter with capitalization + length bias
# ─────────────────────────────
def good_enhanced(s):
    if not s:
        return False, None

    words = s.split()

    # IMPROVEMENT 3: Strict 8-12 word sweet spot
    if not (8 <= len(words) <= 12):
        return False, None

    # Stricter repetition
    if len(set(words)) < len(words) * 0.8:
        return False, None

    # Capitalize + punctuation
    s_fixed = s[0].upper() + s[1:]
    if not s_fixed.rstrip().endswith(('.', '!', '?')):
        s_fixed += '.'

    return True, s_fixed


# ─────────────────────────────
# Production generator (clean, no debug)
# ─────────────────────────────
def generate_best(seed, n_candidates=12):
    candidates = []
    seed_words = set(seed.lower().split())
    words_priority = sorted([w for w in seed.lower().split() if len(w) > 3],
                            key=len, reverse=True)

    # Multi-word seeds
    multi_words = []
    seed_lower = seed.lower()
    for pair in ['stack trace', 'monitoring system', 'booking system', 'barcode scanner']:
        if pair in seed_lower:
            multi_words.append(pair)

    for i in range(n_candidates):
        # Multi-word first
        for mw in multi_words:
            try:
                s = model.make_sentence_with_start(mw, strict=False, max_words=12,
                                                   tries=10, test_output=False)
                ok, s_fixed = good_enhanced(s)
                if ok:
                    candidates.append(s_fixed)
            except:
                pass

        # Single-word seeds (higher state_size preserves context better)
        for seed_word in words_priority[:4]:  # One more word
            try:
                s = model.make_sentence_with_start(
                    seed_word, strict=False, max_words=12,
                    tries=20, test_output=False  # More tries for state_size=4
                )
                ok, s_fixed = good_enhanced(s)
                if ok:
                    candidates.append(s_fixed)
                    if len(candidates) >= 3: break
            except:
                pass

        # Regular with strict relevance
        s = model.make_sentence(tries=25, max_words=12, test_output=False, strict=False)
        ok, s_fixed = good_enhanced(s)
        if ok and (set(s.lower().split()) & seed_words):
            candidates.append(s_fixed)

        if len(candidates) >= 5:  # More candidates = better selection
            break

    # Best by seed overlap
    if candidates:
        def score(s):
            seed_score = len(set(s.lower().split()) & seed_words)
            return seed_score * 2 + (1 / (1 + abs(len(s.split()) - 10)))

        return max(candidates, key=score)

    # Graceful fallback
    noun = words_priority[0] if words_priority else "system"
    return f"Build a {noun.title()} that processes data intelligently."


# ─────────────────────────────
# Rest unchanged
# ─────────────────────────────
def generate_coding_idea():
    adj = random.choice(adjectives) if random.random() < 0.9 else ""
    noun = random.choice(nouns)
    verb = random.choice(verbs)
    twist = random.choice(twists) if random.random() < 0.9 else ""
    return adj, noun, verb, twist


def get_description(adj, noun, verb, twist):
    seed = f"{adj} {noun} {verb} {twist}".strip()
    return generate_best(seed)
