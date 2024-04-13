from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def get_tokenizer_and_model(model_name, device, path_to_model):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))
    model = model.to(device)
    return tokenizer, model


class EntityDataset(Dataset):
    def __init__(self,
                 texts: List[str],
                 labels: Optional[List[int]],
                 entities: List[Tuple[int, int, int]],
                 tokenizer,
                 max_len: int = 512,
                 shifts: int = 256):

        self.samples = []

        for i, text in enumerate(texts):
            _, start, end = entities[i]
            entity_text = text[start:end]

            # Структура токенизированного ввода:
            # [CLS] + предложение + [SEP] + сущность + [SEP] + конец предложения + [EOS]
            pre_entity_text = text[:start]
            idx_space_begin = pre_entity_text.find(' ', start - shifts)
            pre_entity_text = pre_entity_text[idx_space_begin + 1:]

            post_entity_text = text[end:]
            idx_space_end = post_entity_text.rfind(' ', 0, shifts)
            if idx_space_end == -1:
                idx_space_end = len(post_entity_text)
            post_entity_text = post_entity_text[:idx_space_end]

            sequence = f"{pre_entity_text} [SEP] {entity_text} [SEP] {post_entity_text}"

            tokenized_input = tokenizer.encode_plus(
                sequence,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )

            input_ids = tokenized_input['input_ids'].squeeze(0)
            attention_mask = tokenized_input['attention_mask'].squeeze(0)

            label = 0 if labels is None else labels[i]
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def get_pad_collate_with_tokenizer(tokenizer):
    def pad_collate(batch):
        input_ids = [sample["input_ids"] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        labels = [sample["label"] for sample in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.LongTensor(labels),
        }
    return pad_collate


def data_to_model_format(messages, entities):
    new_messages = []
    new_entities = []
    sizes = []
    for i, message in enumerate(messages):
        for j in entities[i]:
            new_messages.append(message)
            new_entities.append(j)
        sizes.append(len(entities[i]))
    return new_messages, new_entities, sizes


def predictions_to_ans_format(preds, entities, sizes):
    ans = []
    ind_list = 0
    for i in sizes:
        message_ans = []
        for j in range(i):
            message_ans.append((int(entities[ind_list][0]), float(preds[ind_list])))
            ind_list += 1
        ans.append(message_ans)
    return ans


def get_sentiment(messages, entities, tokenizer, model, device):
    messages, entities, sizes = data_to_model_format(messages, entities)
    test_dataset = EntityDataset(messages, None, entities, tokenizer)
    pad_collate = get_pad_collate_with_tokenizer(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=5, collate_fn=pad_collate, shuffle=False)

    model.eval()
    test_preds = []

    for batch in test_loader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['label'].to(device)

        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        ans = logits[1].detach().to('cpu').numpy()
        for i in range(len(ans)):
            ans_ind = np.argmax(ans[i]).item()
            test_preds.append(ans_ind + 1)
    return predictions_to_ans_format(test_preds, entities, sizes)
