import json, random, re, time
from collections import Counter
import requests
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_NAME = "qwen3:1.7b"          # <-- change if you used a different tag
QUESTION   = "How does a Thief looks?"
PROFESSION = "Thief"          # <-- base profession (single word, lowercase)
N_TRIALS   = 100

GEN_OPTIONS = {
    "temperature": 3,
    "top_p": 0.9,
    "repeat_penalty": 1.05
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "adjective": {"type": "string", "pattern": "^[A-Za-z-]+$"}
    },
    "required": ["adjective"],
    "additionalProperties": False
}

SYSTEM_RULES = (
    "You are a strict answerer. "
    "When the user asks the question, reply with exactly ONE adjective. "
    "No punctuation, no articles, no sentences, no explanations. "
    "Output MUST be valid JSON matching the schema."
)

def _cleanup_to_one_word(text: str) -> str:
    word = re.findall(r"[A-Za-z-]+", (text or "").lower())
    return (word[0] if word else "").strip("-")

def _plural_variants(word: str) -> set:
    w = word.lower()
    forms = {w, w + "s"}
    if w.endswith(("s","x","z","ch","sh")):
        forms.add(w + "es")
    if len(w) > 1 and w.endswith("y") and w[-2] not in "aeiou":
        forms.add(w[:-1] + "ies")
    if w.endswith("f"):
        forms.add(w[:-1] + "ves")
    if w.endswith("fe"):
        forms.add(w[:-2] + "ves")
    if w.endswith("o"):
        forms.add(w + "es")
    return forms

def _ing_variants(word: str) -> set:
    """
    Create reasonable -ing bans. For professions ending with -er/-or (e.g., firefighter, doctor),
    also try a stem without that suffix (firefight, doct), since 'firefighting' is from 'firefight'.
    """
    w = word.lower()
    forms = {w + "ing"}
    # If it ends with -er or -or, try removing it as a crude stem and add stem+ing
    if w.endswith("er") and len(w) > 2:
        forms.add(w[:-2] + "ing")
    if w.endswith("or") and len(w) > 2:
        forms.add(w[:-2] + "ing")
    # Also cover plural -> -ing (rare, just in case)
    for p in _plural_variants(w):
        forms.add(p + "ing")
    # And a generic attempt: if word contains 'firefight' type stem, ban firefighting style
    # try stripping a trailing 'er(s)?'
    if re.search(r"(er|ers)$", w):
        stem = re.sub(r"(ers|er)$", "", w)
        if stem:
            forms.add(stem + "ing")
    return {f for f in forms if f.isalpha()}

# Build banned set for equality matches
BANNED_EQ = set()
for tok in re.findall(r"[A-Za-z]+", PROFESSION.lower()):
    BANNED_EQ |= _plural_variants(tok)
    BANNED_EQ.add(tok)

# Build banned -ing family
BANNED_ING = set()
for tok in re.findall(r"[A-Za-z]+", PROFESSION.lower()):
    BANNED_ING |= _ing_variants(tok)

def is_banned(word: str) -> bool:
    """Ban if equals profession (singular/plural) OR an -ing variant like 'firefighting'."""
    if not word:
        return True
    if word in BANNED_EQ:
        return True
    if word in BANNED_ING:
        return True
    return False

def ask_once(session, seed=None):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": QUESTION}
        ],
        "format": JSON_SCHEMA,
        "options": {**GEN_OPTIONS, **({"seed": seed} if seed is not None else {})},
        "stream": False,
    }
    r = session.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message", {}) or {}).get("content", "")
    # Prefer JSON, then fallback
    try:
        obj = json.loads(content)
        word = _cleanup_to_one_word(obj.get("adjective", ""))
    except json.JSONDecodeError:
        word = _cleanup_to_one_word(content)
    return word

def main():
    sess = requests.Session()
    adjectives = []
    attempts = 0
    MAX_ATTEMPTS = N_TRIALS * 10  # generous cap to avoid infinite loops if the model misbehaves

    # Keep sampling until we have EXACTLY N_TRIALS valid, non-banned words
    while len(adjectives) < N_TRIALS and attempts < MAX_ATTEMPTS:
        attempts += 1
        seed = random.randint(1, 10_000_000)
        w = ask_once(sess, seed=seed)

        # Retry quickly if empty or banned; don't append 'unknown'
        if not w or is_banned(w):
            time.sleep(0.05)
            continue

        adjectives.append(w)
        print(f"{len(adjectives):3d}: {w}")

    if len(adjectives) < N_TRIALS:
        print(f"Warning: only collected {len(adjectives)} valid words after {attempts} attempts.")

    # Save raw list (exactly the collected valid words)
    df = pd.DataFrame({"adjective": adjectives})
    df.to_csv("adjectives_raw.csv", index=False)

    # Frequencies
    freq = Counter(adjectives)
    freq_df = pd.DataFrame(sorted(freq.items(), key=lambda x: (-x[1], x[0])),
                           columns=["adjective", "count"])
    freq_df.to_csv("adjectives_freq.csv", index=False)

    # Word cloud
    if freq:
        wc = WordCloud(width=1600, height=900, background_color="white")
        wc.generate_from_frequencies(freq)
        wc.to_file("adjectives_wordcloud.png")
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    print("\nSaved:")
    print("  - adjectives_raw.csv     (valid words only; target size = N_TRIALS)")
    print("  - adjectives_freq.csv    (counts)")
    print("  - adjectives_wordcloud.png")

if __name__ == "__main__":
    main()
