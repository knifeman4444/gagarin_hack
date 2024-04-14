import json
import pathlib
import typing as tp
import torch

import final_solution


PATH_TO_TEST_DATA = pathlib.Path("data") / "test_texts.json"
PATH_TO_OUTPUT_DATA = pathlib.Path("results") / "output_scores.json"


def load_data(path: pathlib.PosixPath = PATH_TO_TEST_DATA) -> tp.List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_data(data, path: pathlib.PosixPath = PATH_TO_OUTPUT_DATA):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def main():
    texts = load_data()

    # Хотим работать с cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Параметры решения
    model_name = 'adorkin/xlm-roberta-en-ru-emoji'
    path_to_model = 'models/adorkin_xlm-roberta-en-ru-emoji_0.7138.pth'

    # Загрузка токенайзера и модели
    tokenizer, model = final_solution.bert_sentiment.get_tokenizer_and_model(model_name,
                                                                             device,
                                                                             path_to_model)

    scores = final_solution.solution.score_texts(texts, tokenizer, model, device)
    save_data(scores)


if __name__ == '__main__':
    main()
