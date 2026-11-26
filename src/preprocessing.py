"""NLP text cleaning, emoji conversion, slang normalization, and stopword removal."""

from functools import lru_cache
import re
import string
from typing import Set

import nltk
from nltk.corpus import stopwords

from src.config import EMOJI_DICT, ENGLISH_STOPWORDS, IMPORTANT_WORDS, SLANG_DICT

_URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+")
_MENTION_HASHTAG_PATTERN = re.compile(r"@\w+|#\w+")
_DIGIT_PATTERN = re.compile(r"\d+")
_NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7f]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_PUNCT_TRANS = str.maketrans("", "", string.punctuation)


@lru_cache(maxsize=1)
def get_stopwords() -> Set[str]:
    """Load Indonesian stopwords merged with basic English stopwords, retaining negation words."""
    try:
        indo_stopwords = set(stopwords.words("indonesian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        indo_stopwords = set(stopwords.words("indonesian"))

    combined = indo_stopwords | ENGLISH_STOPWORDS
    return combined - IMPORTANT_WORDS


def clean_and_normalize_text(text: str) -> str:
    """Run the full NLP preprocessing pipeline on a raw review string.

    Steps:
        1. Convert sentiment-bearing emojis into text tokens before stripping non-ASCII.
        2. Case fold, remove URLs, mentions/hashtags, digits, punctuation, and non-ASCII.
        3. Normalize Indonesian slang & code-mixing tokens (handling multi-word expansions).
        4. Remove stopwords while keeping sentiment/negation keywords and words with len > 1.
    """
    if text is None:
        return ""

    processed = str(text)

    # 1. Convert emojis before stripping non-ASCII characters
    for emoji_char, replacement_word in EMOJI_DICT.items():
        if emoji_char in processed:
            processed = processed.replace(emoji_char, replacement_word)

    # 2. Case folding & regex cleaning
    processed = processed.lower()
    processed = _URL_PATTERN.sub("", processed)
    processed = _MENTION_HASHTAG_PATTERN.sub("", processed)
    processed = _DIGIT_PATTERN.sub("", processed)
    processed = processed.translate(_PUNCT_TRANS)
    processed = _NON_ASCII_PATTERN.sub("", processed)
    processed = _WHITESPACE_PATTERN.sub(" ", processed).strip()

    if not processed:
        return ""

    # 3 & 4. Slang normalization + stopword removal
    stopword_set = get_stopwords()
    normalized_words: list[str] = []

    for token in processed.split():
        normalized = SLANG_DICT.get(token, token)
        for sub_token in normalized.split():
            if sub_token not in stopword_set and len(sub_token) > 1:
                normalized_words.append(sub_token)

    return " ".join(normalized_words)
