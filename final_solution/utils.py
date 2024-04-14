import pymystem3
import re

stemmer = pymystem3.Mystem()

def normalize_text(text: str) -> str:
    """
    Normalize text (remove punctuation, lowercase, lemmatize)
    Args:
        text (str): text to normalize
    Returns:
        str: normalized text
    """
    if text is None:
        return ''

    text = re.sub(r'[^\w\s]', '', text).lower()
    text = ' '.join(text.split())
    words = stemmer.lemmatize(text)
    return ''.join(words).strip()
