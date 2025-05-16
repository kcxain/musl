from dataclasses import dataclass
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from trans.utils.template import template_dict
from trans.utils.prompts import CUDA_CPP_TRANSLATE_SYSTEM, CUDA_CPP_TRANSLATE_USER, CUDA_CPP_TRANSLATE_TRAIN_SYSTEM
from trans.utils.io import rank0_print, load_tok_file, load_jsonlines, dump_json
from typing import Optional
from pydantic import BaseModel


class TransDirection:
    source: str
    target: str

    def __init__(self, source: Optional[str] = None, target: Optional[str] = None, **kwargs):
        trans_pair = {
            "CUDA": "CPP",
            "CPP": "CUDA",
        }
        if source:
            self.source = source.upper()
            self.target = trans_pair[source.upper()]
        elif target:
            self.target = target.upper()
            self.source = trans_pair[target.upper()]
        else:
            raise ValueError("Either source or target must be provided.")
        

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: PreTrainedTokenizer,
        direction: TransDirection,
        data_args
    ):
        self.tokenizer = tokenizer
        self.direction = direction
        rank0_print("Formatting inputs...")
        self.tokenizer = tokenizer
        self.raw_data = raw_data

        self.data = []
        for raw_line in self.raw_data:
            ins = raw_line
            assert direction.source in ins or f'{direction.source}-gen' in ins
            input_code = ins[f'{direction.source}-gen'] if f'{direction.source}-gen' in ins else ins[direction.source]
            output_code = ins[direction.target]
            system = CUDA_CPP_TRANSLATE_TRAIN_SYSTEM.format(obj=direction)
            if input_code == "":
                continue
            self.data.append({
                "instruction": system,
                "input": input_code,
                "output": output_code
            })
        if data_args.max_samples is not None:
            self.data = self.data[:data_args.max_samples]
        rank0_print(f"Total number of examples: {len(self.data)}")
        rank0_print(f'Example: \n{self.data[0]}')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

def get_train_dataset(direction: TransDirection, data_args):
    train_dataset = load_dataset(
        "json",
        data_files=data_args.data_path,
        split="train",
    )
    train_dataset = train_dataset.remove_columns(["input", "output"])
    train_dataset = train_dataset.rename_column(f'{direction.source}-gen', 'input')
    train_dataset = train_dataset.rename_column(f'{direction.target}', 'output')
    return train_dataset
    
def get_eval_dataset(direction: TransDirection, data_args):
    eval_dataset = load_dataset(
        "json",
        data_files=data_args.eval_data_dir,
        split="train",
    )
    eval_dataset = eval_dataset.rename_column(f'{direction.source}', 'input')
    eval_dataset = eval_dataset.rename_column(f'{direction.target}', 'output')
    return eval_dataset


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, direction: TransDirection, data_args) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    rank0_print("Loading data...")

    train_dataset = get_train_dataset(
        direction=direction, data_args=data_args
    )

    if data_args.eval_data_dir:
        # eval_data = load_tok_file(data_args.eval_data_dir, direction.source, mode='test')
        eval_dataset = get_eval_dataset(
            direction=direction, data_args=data_args
        )
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

    
class EvalDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        source,
        target
    ):
        self.data = data
        self.source = source.upper()
        self.target = target.upper()
        self.template = template_dict['llama3']
        self.system = CUDA_CPP_TRANSLATE_SYSTEM.format(obj=self)
        self.tokenizer = tokenizer
        print(f'datasets: {len(self)}')
        print(f'task: {self.source} -> {self.target}')
        print(self[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx][self.source]
        label = self.data[idx][self.target]
        user_prompt = CUDA_CPP_TRANSLATE_USER.format(obj=self, content=ins)
        message = [
            {"role": "system","content": self.system},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        # prompt = self.template.build_prompt(self.system, ins)
        return prompt, label

    def get_all(self):
        return [self[i] for i in range(len(self))]
    
    def get_raw(self):
        return self.data

class EvalSFTDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        source,
        target
    ):
        self.data = data
        self.source = source.upper()
        self.target = target.upper()
        self.system = CUDA_CPP_TRANSLATE_TRAIN_SYSTEM.format(obj=self)
        self.tokenizer = tokenizer
        print(f'datasets: {len(self)}')
        print(f'task: {self.source} -> {self.target}')
        print(self[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx][self.source]
        label = self.data[idx][self.target]
        user_prompt = ins
        message = [
            {"role": "system","content": self.system},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        # prompt = self.template.build_prompt(self.system, ins)
        return prompt, label

    def get_all(self):
        return [self[i] for i in range(len(self))]
    
    def get_raw(self):
        return self.data

class CollateFnWithTokenization:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_len: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        outputs = self.tokenizer(
            batch,
            return_tensors="pt",
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        return outputs

class InferenceBTDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        source,
        target
    ):
        self.data = data
        self.source = source.upper()
        self.target = target.upper()
        self.template = template_dict['llama3']
        self.system = CUDA_CPP_TRANSLATE_TRAIN_SYSTEM.format(obj=self)
        self.tokenizer = tokenizer
        print(f'datasets: {len(self)}')
        print(f'task: {self.source} -> {self.target}')
        print(self[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx][self.source]
        user_prompt = ins
        message = [
            {"role": "system","content": self.system},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        # prompt = self.template.build_prompt(self.system, ins)
        return prompt

    def get_all(self):
        return [self[i] for i in range(len(self))]
    
    def get_raw(self):
        return self.data

class InferenceBTConstrainDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        source,
        target
    ):
        self.data = data
        self.source = source.upper()
        self.target = target.upper()
        # self.template = template_dict['llama3']
        self.system = CUDA_CPP_TRANSLATE_SYSTEM.format(obj=self)
        self.tokenizer = tokenizer
        print(f'datasets: {len(self)}')
        print(f'task: {self.source} -> {self.target}')
        print(self[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins = self.data[idx][self.source]
        user_prompt = CUDA_CPP_TRANSLATE_USER.format(obj=self, content=ins)
        message = [
            {"role": "system","content": self.system},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        # prompt = self.template.build_prompt(self.system, ins)
        return prompt

    def get_all(self):
        return [self[i] for i in range(len(self))]
    
    def get_raw(self):
        return self.data
