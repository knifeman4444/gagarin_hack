import typing as tp
from . import dataloader
from tqdm import tqdm

Entity = tp.Tuple[int, int, int]  # (entity_id, (start, end))


def find_entities(messages: tp.Iterable[str]) -> tp.List[tp.List[Entity]]:
    """
    Find entities in messages using dictionary of synonims
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)
    Returns:
        tp.List[tp.List[Entity]]: for any messages returns list of entities
    """

    result = []
    for message in tqdm(messages):
        entities = []
        for id, syns in dataloader.synonyms_dict.items():
            for syn in syns:
                start = message.lower().find(syn)
                if start != -1:
                    end = start + len(syn)
                    entities.append((id, start, end))
                    break
        result.append(entities)
    return result





