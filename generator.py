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

def clean_text(text: str) -> str:
    allowed = set(string.ascii_letters + string.digits + ' .!?\n')
    return ''.join(c if c in allowed else ' ' for c in text).lower()

cleaned_text = clean_text(raw_text)

# ─────────────────────────────────────────────────────────────────────────────
# Build higher-order Markov models
# ─────────────────────────────────────────────────────────────────────────────
word_model = markovify.Text(cleaned_text, state_size=5)  # higher-order for coherence
char_model = markovify.NewlineText(cleaned_text, state_size=6, well_formed=False)

# ─────────────────────────────────────────────────────────────────────────────
# Post-processing for punctuation and capitalization
# ─────────────────────────────────────────────────────────────────────────────
def polish_sentence(sentence: str) -> str:
    if not sentence:
        return ""
    sentence = re.sub(r'\s+([.,!?])', r'\1', sentence)  # fix spacing
    sentence = sentence[0].upper() + sentence[1:]       # capitalize first letter
    if sentence[-1] not in '.!?':
        sentence += '.'
    return sentence

# ─────────────────────────────────────────────────────────────────────────────
# Generate a sentence aligned with the idea
# ─────────────────────────────────────────────────────────────────────────────
def generate_text_idea(idea_seed: str, max_words: int = 35, num_candidates: int = 5) -> str:
    """
    Generate text aligned with the idea without using a strict starting word.
    """
    # Generate multiple candidates
    candidates = []
    for _ in range(num_candidates):
        sentence = word_model.make_sentence(tries=100, max_words=max_words)
        if not sentence:
            sentence = char_model.make_sentence(tries=100, max_words=max_words)
        if sentence:
            candidates.append(polish_sentence(sentence))

    # Rerank candidates by Markov log probability (internal coherence)
    scored = [(c, word_model.log_probability(c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_candidates = [c for c, _ in scored]

    # Return the top candidate
    return best_candidates[0] if best_candidates else ""

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
# Measure similarity using TF-IDF
# ─────────────────────────────────────────────────────────────────────────────
def measure_similarity(idea: str, candidates: list[str]) -> list[float]:
    all_texts = [idea] + candidates
    vectorizer = TfidfVectorizer().fit(all_texts)
    vectors = vectorizer.transform(all_texts)
    sim_matrix = cosine_similarity(vectors[0:1], vectors[1:])
    return sim_matrix.flatten().tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Generate description aligned with idea, avoid duplicates
# ─────────────────────────────────────────────────────────────────────────────
def get_description(adj: str, noun: str, verb: str, twist: str, existing: list[str] | None = None) -> str:
    idea_seed = f"{adj} {noun} {verb} {twist}".strip()
    sentences = []

    # Generate multiple candidate sentences
    candidates = [generate_text_idea(idea_seed) for _ in range(5)]

    # Filter candidates that are too similar to existing ideas
    if existing:
        filtered = []
        for c in candidates:
            sims = measure_similarity(c, existing)
            if all(s < 0.6 for s in sims):  # uniqueness threshold
                filtered.append(c)
        candidates = filtered or candidates

    # Pick up to 3 sentences
    sentences.extend(candidates[:3])
    return ' '.join(sentences)

# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    adj, noun, verb, twist = generate_coding_idea()
    description = get_description(adj, noun, verb, twist, existing=[])
    print("Idea:", adj, noun, verb, twist)
    print("Description:", description)
