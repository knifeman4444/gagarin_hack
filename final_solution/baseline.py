import typing as tp
from .dataloader import synonyms_dict, tickers
from .utils import normalize_text
from tqdm import tqdm
import ahocorasick
import re

Entity = tp.Tuple[int, int, int]  # (entity_id, start, end)

tickers_automaton = ahocorasick.Automaton()
for id, ticker in tickers.items():
    tickers_automaton.add_word(ticker, (id, ticker))
tickers_automaton.make_automaton()

synonyms_automaton = ahocorasick.Automaton()
for id, syns in synonyms_dict.items():
    for syn in syns:
        synonyms_automaton.add_word(syn, (id, syn))
synonyms_automaton.make_automaton()


def find_entities(messages: tp.Iterable[str]) -> tp.List[tp.List[Entity]]:
    """
    Find entities in messages using dictionary of synonyms
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)
    Returns:
        tp.List[tp.List[Entity]]: for any messages returns list of entities (might repeat)
    """

    result = []
    for message in tqdm(messages):
        message = message.lower()
        initial_message = message

        def replace(match):
            return ' ' * len(match.group(0))

        message = re.sub(r'\b[^\w\s]+\b|\B[^\w\s]+\B', replace, message)

        spaces = [-1]
        for i, c in enumerate(message):
            if c.isspace() and i != 0 and not message[i - 1].isspace():
                spaces.append(i)
        spaces.append(len(message))
        word_bounds = []
        for i in range(1, len(spaces)):
            word_bounds.append((spaces[i - 1] + 1, spaces[i] - 1))

        message = normalize_text(message)
        word_indices = [0] * len(message)
        for i in range(1, len(message)):
            word_indices[i] = word_indices[i - 1] + (message[i - 1].isspace() and not message[i].isspace())

        def get_real_bounds(start, end):
            start = word_bounds[word_indices[start]][0]
            end = word_bounds[word_indices[end]][1]

            while not (initial_message[start].isalpha() or initial_message[start].isdigit()):
                start += 1
            while not (initial_message[end].isalpha() or initial_message[end].isdigit()):
                end -= 1

            return start, end

        entities = []

        entities_used = set()
        for end, (id, ticker) in tickers_automaton.iter(message):
            start = end - len(ticker) + 1
            start, end = get_real_bounds(start, end)
            entities.append((id, start, end))
            entities_used.add(id)

        for end, (id, syn) in synonyms_automaton.iter(message):
            if id in entities_used:
                continue
            start = end - len(syn) + 1
            if start != 0 and not message[start - 1].isspace():
                continue
            if end != len(message) - 1 and not message[end + 1].isspace():
                continue

            start, end = get_real_bounds(start, end)

            entities.append((id, start, end))
            entities_used.add(id)

        result.append(entities)

    return result
