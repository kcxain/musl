from typing import Dict, Optional, Union
from numpy import ndarray as NDArray
import torch
import numpy as np
import evaluate
from transformers import PreTrainedTokenizer, EvalPrediction
from dataclasses import dataclass
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def numpify(inputs: Union["NDArray", "torch.Tensor"]) -> "NDArray":
    r"""
    Casts a torch tensor or a numpy array to a numpy array.
    """
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu()
        if inputs.dtype == torch.bfloat16:  # numpy does not support bfloat16 until 1.21.4
            inputs = inputs.to(torch.float32)

        inputs = inputs.numpy()

    return inputs


def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer):
    preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
    bleu_metric = evaluate.load("trans/evaluation/bleu")
    codebleu_metric = evaluate.load("trans/evaluation/codebleu")
    preds = np.where(preds != IGNORE_TOKEN_ID, preds, tokenizer.pad_token_id)
    labels = np.where(labels != IGNORE_TOKEN_ID, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    codebleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    bleu_dict = bleu_metric.compute(),
    codebleu_dict = codebleu_metric.compute(lang="cpp", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)

    return bleu_dict.update(codebleu_dict)

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels