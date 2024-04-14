import typing as tp
import final_solution


EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[
    EntityScoreType
]  # list of entity scores,
#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]


def score_texts(
    messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 2048 for m in messages]) # all messages are shorter than 2048 characters
    """

    tokenizer, model, device = args
    entities = final_solution.baseline.find_entities(messages)
    sentiments = final_solution.bert_sentiment.get_sentiment(messages, entities, tokenizer, model, device)
    return sentiments
