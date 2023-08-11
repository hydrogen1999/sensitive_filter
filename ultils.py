from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients
import unidecode
import re
import torch
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def remove_accent(sentence):
    return unidecode.unidecode(sentence)


class CheckBadWords:
    def __init__(self, badwords, blockwords):
        self.pattern1 = '|'.join(
            [r"\b" + badword + r"\b" for badword in badwords])
        self.pattern2 = '|'.join(blockwords)

    def __call__(self, str_):
        if re.findall(self.pattern1, str_):
            return 1
        elif re.findall(self.pattern2, str_):
            return 1
        else:
            return 0


def infer(text, tokenizer, model, max_len=120):
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    probabilities = F.softmax(output, dim=-1)
    probabilities = torch.max(probabilities, dim=1)
    _, y_pred = torch.max(output, dim=1)
    output = [probabilities, y_pred]
    return output