"""Tokenization, stopword removal, and stemming utilities."""

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Shared tokenizer (documents + queries) to keep processing consistent
TOKENIZER = RegexpTokenizer(
    r"(?:[A-Za-z]\.)+"            # abbreviations like U.S.A.
    r"|(?:[A-Za-z]+[\-@]\d+(?:\.\d+)?)"  # words with - or @ and numbers like A-123
    r"|\d+(?:[.,-]\d+)*%?"        # numbers, decimals, percentages
    r"|[A-Za-z]+"                  # regular words
)

_STOPWORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


def preprocess(text: str) -> list[str]:
    """Normalize text into a list of stemmed tokens.

    Steps (shared for docs and queries): tokenization → lowercase → stopword removal → Porter stemming.
    """

    tokens = TOKENIZER.tokenize(text)
    lowered = (t.lower() for t in tokens)
    filtered = (t for t in lowered if t not in _STOPWORDS)
    stemmed = [_STEMMER.stem(t) for t in filtered if t]
    return stemmed


def preprocess_mapping(text_by_id: dict[int, str]) -> dict[int, list[str]]:
    """Apply preprocessing to a mapping of ids → raw text."""

    return {item_id: preprocess(text) for item_id, text in text_by_id.items()}
