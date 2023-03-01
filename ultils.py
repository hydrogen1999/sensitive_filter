from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients
import unidecode
import re
import torch

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
    _, y_pred = torch.max(output, dim=1)
    return y_pred


def construct_input_and_baseline(text, tokenizer):

    max_length = 510
    baseline_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token_id = tokenizer.cls_token_id

    text_ids = tokenizer.encode(text,
                                max_length=max_length,
                                truncation=True,
                                add_special_tokens=False)

    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    token_list = tokenizer.convert_ids_to_tokens(input_ids)

    baseline_input_ids = [
        cls_token_id
    ] + [baseline_token_id] * len(text_ids) + [sep_token_id]
    return torch.tensor([input_ids],
                        device='cpu'), torch.tensor([baseline_input_ids],
                                                    device='cpu'), token_list


def summarize_attributions(attributions):

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    return attributions


def interpret_text(text, model, tokenizer, true_class='unknow'):

    input_ids, baseline_input_ids, all_tokens = construct_input_and_baseline(
        text, tokenizer)

    # Define model output
    def model_output(inputs):
        return model(inputs)[0]

    # Define model input
    model_input = model.bert.embeddings
    lig = LayerIntegratedGradients(model_output, model_input)
    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=baseline_input_ids,
                                        return_convergence_delta=True,
                                        internal_batch_size=1)
    attributions_sum = summarize_attributions(attributions)
    print(attributions_sum)
    score_vis = viz.VisualizationDataRecord(
        word_attributions=attributions_sum,
        pred_prob=torch.max(model(input_ids)[0]),
        pred_class=torch.argmax(model(input_ids)[0]).numpy(),
        true_class=true_class,
        attr_class=text,
        attr_score=attributions_sum.sum(),
        raw_input_ids=all_tokens,
        convergence_score=delta)
    viz.visualize_text([score_vis])
