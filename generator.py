import markovify
import re
import string
import random
from word_lists import adjectives, nouns, verbs, twists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# Load and clean corpus
# ─────────────────────────────────────────────────────────────────────────────
with open("seed_text.py", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Clean text: lowercase, remove weird punctuation but keep sentence end markers
def clean_text(text: str) -> str:
    allowed = set(string.ascii_letters + string.digits + ' .!?\n')
    return ''.join(c if c in allowed else ' ' for c in text).lower()

cleaned_text = clean_text(raw_text)

# ─────────────────────────────────────────────────────────────────────────────
# Build word-level and character-level Markov models (higher-order)
# ─────────────────────────────────────────────────────────────────────────────
word_model = markovify.Text(cleaned_text, state_size=4)
char_model = markovify.NewlineText(cleaned_text, state_size=6, well_formed=False)

# ─────────────────────────────────────────────────────────────────────────────
# Generate text with optional seed for context
# ─────────────────────────────────────────────────────────────────────────────
def generate_text(seed: str | None = None, word_weight=0.7, char_weight=0.3, max_words=35) -> str:
    sentence = None

    if seed:
        words = seed.split()
        # Try each word in the seed as starting point until one works
        for i in range(len(words)):
            try:
                sentence = word_model.make_sentence_with_start(
                    beginning=words[i], strict=False, max_words=max_words
                )
                if sentence:
                    break
            except (KeyError, markovify.text.ParamError):
                continue

    # Fallback: word-level or character-level generation
    if not sentence or random.random() < char_weight:
        sentence = word_model.make_sentence(tries=100, max_words=max_words)
        if not sentence:
            sentence = char_model.make_sentence(tries=100, max_words=max_words)

    # Capitalize and ensure punctuation
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in '.!?':
            sentence += '.'

    return sentence or ""

# ─────────────────────────────────────────────────────────────────────────────
# Generate coding idea tuple
# ─────────────────────────────────────────────────────────────────────────────
def generate_coding_idea() -> tuple[str, str, str, str]:
    adj = random.choice(adjectives) if random.random() < 0.9 else ""
    noun = random.choice(nouns)
    verb = random.choice(verbs)
    twist = random.choice(twists) if random.random() < 0.9 else ""
    return adj, noun, verb, twist

# ─────────────────────────────────────────────────────────────────────────────
# Measure similarity between ideas using TF-IDF cosine similarity
# ─────────────────────────────────────────────────────────────────────────────
def measure_similarity(idea: str, candidates: list[str]) -> list[float]:
    """
    Return cosine similarity between one idea and multiple candidate ideas.
    """
    all_texts = [idea] + candidates
    vectorizer = TfidfVectorizer().fit(all_texts)
    vectors = vectorizer.transform(all_texts)
    sim_matrix = cosine_similarity(vectors[0:1], vectors[1:])
    return sim_matrix.flatten().tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Generate description with multiple candidates and pick least similar one
# ─────────────────────────────────────────────────────────────────────────────
def get_description(adj: str, noun: str, verb: str, twist: str, existing: list[str] | None = None) -> str:
    """Return a 1-3 sentence description using Markov generator and avoid similar existing ideas."""
    seed = f"{adj} {noun} {verb} {twist}".strip()
    sentences = []

    # Generate multiple candidate sentences
    candidates = [generate_text(seed) for _ in range(5)]

    # If existing ideas are provided, filter out too-similar ones
    if existing:
        filtered = []
        for c in candidates:
            sims = measure_similarity(c, existing)
            if all(s < 0.6 for s in sims):  # threshold for uniqueness
                filtered.append(c)
        candidates = filtered or candidates

    # Pick up to 3 candidates for description
    sentences.extend(candidates[:3])
    return ' '.join(sentences)