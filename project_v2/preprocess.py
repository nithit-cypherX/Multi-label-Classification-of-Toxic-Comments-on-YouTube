"""
preprocess.py
-------------
Dual-mode text preprocessing pipeline.

  classical mode  →  heavy cleaning (lemmatize, stopwords, slang expand)
                     Used for: Logistic Regression + TF-IDF
  bert mode       →  light cleaning (preserve grammar for attention heads)
                     Used for: DistilBERT fine-tuning
"""

import re
import unicodedata
import contractions
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ── NLTK downloads (safe to call multiple times) ───────────────────────────────
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else resource)
    except LookupError:
        nltk.download(resource, quiet=True)

# ── Constants ──────────────────────────────────────────────────────────────────
_LEMMATIZER = WordNetLemmatizer()

# Words that flip meaning — never remove these as "stopwords"
_NEGATION_WORDS = {
    'no', 'not', 'never', 'nobody', 'nothing', 'nowhere',
    'neither', 'nor', 'none', 'cannot', 'without',
}

_STOP_WORDS = stopwords.words('english')
_STOP_WORDS = set(w for w in _STOP_WORDS if w not in _NEGATION_WORDS)

# Common internet slang → expanded form
_SLANG_MAP = {
    'u': 'you', 'ur': 'your', 'r': 'are', 'y': 'why',
    'lol': 'laugh out loud', 'lmao': 'laugh', 'rofl': 'laugh',
    'omg': 'oh my god', 'wtf': 'what the fuck', 'wth': 'what the hell',
    'stfu': 'shut the fuck up', 'gtfo': 'get the fuck out',
    'tbh': 'to be honest', 'imo': 'in my opinion', 'imho': 'in my humble opinion',
    'smh': 'shaking my head', 'fyi': 'for your information',
    'afaik': 'as far as i know', 'irl': 'in real life',
    'ngl': 'not gonna lie', 'ikr': 'i know right',
    'idk': 'i do not know', 'idc': 'i do not care',
    'brb': 'be right back', 'btw': 'by the way',
    'asap': 'as soon as possible', 'bc': 'because',
    'pls': 'please', 'plz': 'please', 'thx': 'thanks', 'ty': 'thank you',
    'cya': 'see you', 'kk': 'okay', 'np': 'no problem',
}

# Intentional obfuscation patterns (toxic users bypass filters this way)
_OBFUSCATION_PATTERNS = [
    (r'\bf[\*@#!]ck', 'fuck'),
    (r'\bs[\*@#!]it', 'shit'),
    (r'\ba[\*@#!]+hole', 'asshole'),
    (r'\bb[\*@#!]tch', 'bitch'),
    (r'\bd[\*@#!]mn', 'damn'),
    (r'\bc[\*@#!]nt', 'cunt'),
    (r'\bh[\*@#!]te', 'hate'),
    (r'\bk[\*@#!]ll', 'kill'),
    (r'\bst[\*@#!]pid', 'stupid'),
    (r'\bidi[\*@#!]t', 'idiot'),
]


# ── Helper functions ───────────────────────────────────────────────────────────

def _normalise_apostrophes(text: str) -> str:
    """Replace curly quotes and common missing apostrophes."""
    text = text.replace('\u2019', "'").replace('\u2018', "'").replace('`', "'")
    # Fix common missing apostrophes
    text = re.sub(r'\b(dont|cant|wont|didnt|doesnt|isnt|arent|wasnt|werent|havent|hasnt|hadnt|wouldnt|couldnt|shouldnt|mustnt)\b',
                  lambda m: m.group(0)[:-1] + "'" + m.group(0)[-1] if len(m.group(0)) > 4 else m.group(0),
                  text, flags=re.IGNORECASE)
    return text


def _handle_obfuscation(text: str) -> str:
    """Decode intentional censoring: f*ck → fuck."""
    for pattern, replacement in _OBFUSCATION_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _handle_emojis(text: str) -> str:
    """Replace emoji characters with EMOJITAG token (preserves signal without noise)."""
    emoji_pattern = re.compile(
        '[\U00010000-\U0010ffff'
        '\U0001F600-\U0001F64F'
        '\U0001F300-\U0001F5FF'
        '\U0001F680-\U0001F6FF'
        '\U0001F1E0-\U0001F1FF'
        '\u2600-\u26FF\u2700-\u27BF]+',
        flags=re.UNICODE
    )
    return emoji_pattern.sub(' EMOJITAG ', text)


def _normalize_slang(text: str) -> str:
    """Expand internet slang using lookup table."""
    tokens = text.split()
    return ' '.join(_SLANG_MAP.get(t.lower(), t) for t in tokens)


def _get_wordnet_pos(treebank_tag: str):
    """Convert Penn Treebank POS tag to WordNet POS."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def _lemmatize(tokens: list[str]) -> list[str]:
    """POS-tag then lemmatize. Avoids 'better' → 'better' (wrong) vs 'good' (correct)."""
    pos_tags = pos_tag(tokens)
    return [
        _LEMMATIZER.lemmatize(word, _get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]


# ── Public API ─────────────────────────────────────────────────────────────────

def preprocess(text: str, mode: str = 'classical') -> str:
    """
    Clean a single comment string.

    Args:
        text:  raw comment text
        mode:  'classical' (heavy, for LR/TF-IDF) or 'bert' (light, for DistilBERT)

    Returns:
        Cleaned string ready for vectorisation or tokenisation.
    """
    if not isinstance(text, str) or not text.strip():
        return ''

    # ── Shared steps (both modes) ──────────────────────────────────────────────
    text = _normalise_apostrophes(text)
    text = _handle_emojis(text)
    text = _handle_obfuscation(text)
    text = contractions.fix(text)

    if mode == 'bert':
        # DistilBERT needs natural sentence structure — minimal touch
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)   # remove URLs
        text = re.sub(r'\S+@\S+', '', text)                    # remove emails
        text = re.sub(r'[^\w\s]', ' ', text)                   # keep word chars
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    # ── Classical-only steps ──────────────────────────────────────────────────
    text = _normalize_slang(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    tokens = _lemmatize(tokens)

    return ' '.join(tokens)


def preprocess_batch(texts: list[str], mode: str = 'classical') -> list[str]:
    """Apply preprocess() to a list of strings."""
    return [preprocess(t, mode=mode) for t in texts]
