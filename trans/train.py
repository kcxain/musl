import os

import transformers
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer

from trans.dataset import make_supervised_data_module, TransDirection
from trans.utils.config import DataArguments, ModelArguments, TrainingArguments
from trans.utils.io import safe_save_model_for_hf_trainer
from trans.evaluation.metrics import compute_metrics, preprocess_logits_for_metrics
from trans.utils.prompts import CUDA_CPP_TRANSLATE_TRAIN_SYSTEM

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
    direction = TransDirection(source=data_args.source, target=data_args.target)

    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, direction=direction, data_args=data_args)

    def formatting_prompts_func(example):
        message = [
            {"role": "system","content": CUDA_CPP_TRANSLATE_TRAIN_SYSTEM.format(obj=direction)},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = False)
        return [text]

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, max_seq_length=model_args.max_length, args=training_args, formatting_func=formatting_prompts_func, data_collator=collator, **data_module, 
    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()