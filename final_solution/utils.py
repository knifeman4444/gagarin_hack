import pymorphy3
import re

morph = pymorphy3.MorphAnalyzer()


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
    words = text.split()
    words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(words)