import pytest
import torch

import final_solution
from final_solution import bert_sentiment


# Хотим работать с cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Параметры решения
model_name = 'google-bert/bert-base-multilingual-cased'
path_to_model = 'models/google-bert_bert-base-multilingual-cased_0.6345.pth'

# Загрузка токенайзера и модели
tokenizer, model = final_solution.bert_sentiment.get_tokenizer_and_model(model_name,
                                                                         device,
                                                                         path_to_model)


def test_empty():
    """If score_texts do not pass this test, it is fine"""

    assert not bool(final_solution.solution.score_texts([], tokenizer, model, device))
    
    # nothing was found
    nothing = final_solution.solution.score_texts([""], tokenizer, model, device)
    assert nothing == [[]]


def test_one_message():
    """Format of answers is important"""
    messages = ["Сбер, он и в Африке Сбер"]
    correct_scores = [[(150, 3.0)]]
    
    assert final_solution.solution.score_texts(messages, tokenizer, model, device) == correct_scores


def test_two_entities_one_message():
    """Order of companies inside one message is not important"""
    messages = ["Сбер, он и в Африке. Тинькофф, он и в Африке Тинькофф"]
    correct_scores = [[(150, 3.0), (225, 3.0)]]
    
    scores = final_solution.solution.score_texts(messages, tokenizer, model, device)

    assert [set(s) == set(cs) for s, cs in zip(scores, correct_scores)]


def test_two_entities_two_messages():
    """"""
    messages = ["Сбер, он и в Африке Сбер", "Тинькофф, он и в Африке Тинькофф"]
    correct_scores = [[(150, 3.0)], [(225, 3.0)]]

    assert final_solution.solution.score_texts(messages, tokenizer, model, device) == correct_scores


def test_large_sequence(N = 10 ** 3):
    """No matter how large N is, score_texts function should work"""
    message = "Сбер, он и в Африке Сбер"
    correct_score = [(150, 3.0)]

    assert final_solution.solution.score_texts([message] * N, tokenizer, model, device) == [correct_score] * N
