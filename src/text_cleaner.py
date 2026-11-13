"""
text_cleaner.py - Cleans text and splits it into words (tokens).

clean_text() : light cleaning for text we store and show (keeps line breaks).
tokenize()   : aggressive cleaning for matching (lowercase, words only).
The SAME tokenize() must be used for documents and for user questions.
"""

import re

# ---------------------------------------------------------------------------
# Patterns for clean_text()
# ---------------------------------------------------------------------------
INVISIBLE_CHARS = re.compile("[\u200b\u200c\u200d\ufeff\x0c]")
EXTRA_SPACES = re.compile(r"[ \t\u00a0]+")
EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
PAGE_NUMBER_LINE = re.compile(
    r"^(page\s+)?\d{1,3}(\s*(of|/)\s*\d{1,3})?$", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Settings for tokenize()
# ---------------------------------------------------------------------------
# IT-specific spellings that should count as the same word when matching.
NORMALIZATIONS = [
    (re.compile(r"\bwi[\s-]?fi\b"), "wifi"),
    (re.compile(r"\be[\s-]?mail\b"), "email"),
    (re.compile(r"\b(?:log|sign)[\s-]?(?:in|on)\b"), "login"),
    (re.compile(r"\b(?:cannot|cant)\b"), "cannot"),
]

WORD_PATTERN = re.compile(r"[a-z0-9]+")

# Very common words that carry no topic information.
# "no" and "not" are deliberately kept: they matter in "no internet".
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "am", "i", "me", "my", "we", "our", "you", "your", "it",
    "its", "they", "them", "their", "this", "that", "these", "those", "to",
    "of", "in", "on", "at", "for", "with", "from", "by", "as", "into", "do",
    "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "may", "might", "how", "what", "why", "when", "where", "which",
    "who", "there", "here", "so", "than", "then", "also", "just", "please",
    "any", "some", "all", "about", "up", "out",
}


def clean_text(text):
    """Remove junk (odd characters, extra spaces, page-number lines) but keep line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = INVISIBLE_CHARS.sub("", text)
    text = EXTRA_SPACES.sub(" ", text)

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if PAGE_NUMBER_LINE.match(line):
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = EXTRA_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def tokenize(text):
    """Turn text into a list of lowercase words, ready for TF-IDF."""
    text = text.lower()
    text = re.sub(r"['\u2019]s\b", "", text)       # phone's -> phone
    text = text.replace("'", "").replace("\u2019", "")  # can't -> cant

    for pattern, replacement in NORMALIZATIONS:
        text = pattern.sub(replacement, text)

    words = WORD_PATTERN.findall(text)
    return [word for word in words if word not in STOP_WORDS]


# ---------------------------------------------------------------------------
# Quick self-check: run with  python -m src.text_cleaner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    messy = (
        "  Wi-Fi   Guide\r\n\r\n\r\n\r\nStep 1:\u00a0Turn  off Wi-Fi.\n"
        "   Page 1 of 2\n\n3\nDone.  "
    )
    cleaned = clean_text(messy)
    print(repr(cleaned))
    assert cleaned == "Wi-Fi Guide\n\nStep 1: Turn off Wi-Fi.\n\nDone."

    result = tokenize("My phone's Wi-Fi can't connect, no internet!")
    print(result)
    assert result == ["phone", "wifi", "cannot", "connect", "no", "internet"]

    result = tokenize("Can't sign in to my e-mail")
    print(result)
    assert result == ["cannot", "login", "email"]

    result = tokenize("The Wi-Fi is connected but there is no internet access")
    print(result)
    assert result == ["wifi", "connected", "no", "internet", "access"]

    print("All text_cleaner checks passed.")