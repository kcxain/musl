import os
import json
from pathlib import Path

import transformers

def rank0_print(*args, **kwargs):
    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(*args, **kwargs)

def load_tok_file(tok_file_dir, source, mode) -> list[str]:
    assert mode in ["train", "test", "valid"]
    source = source.lower()
    if mode == "train":
        tok_file_path = Path(tok_file_dir) / f"{source}.mono.{mode}.tok"
        print(f"Loading {tok_file_path} ...")
        with tok_file_path.open("r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        lines = [
            # 只分割第一个 '|'，规避 '||'
            line.split("|", 1)[1].strip() if line.strip() and "|" in line else line
            for line in lines
        ]
        lines = [line for line in lines if line]
        return lines

    if mode == 'test':
        cpp_file_path = Path(tok_file_dir) / f"cpp.para.{mode}.tok"
        cuda_file_path = Path(tok_file_dir) / f"cuda.para.{mode}.tok"
        with cpp_file_path.open("r", encoding="utf-8") as f:
            cpp_lines = f.read().splitlines()
        with cuda_file_path.open("r", encoding="utf-8") as f:
            cuda_lines = f.read().splitlines()
        assert len(cpp_lines) == len(cuda_lines)
        para_list = [{'CPP': cpp, 'CUDA': cuda} for cpp, cuda in zip(cpp_lines, cuda_lines)]
        return para_list

    else:
        raise NotImplementedError

def dump_jsonlines(obj, filepath, **kwargs):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wt", encoding="utf-8") as fout:
        for d in obj:
            line_d = json.dumps(d, ensure_ascii=False, **kwargs)
            fout.write("{}\n".format(line_d))


def load_jsonlines(filepath, **kwargs):
    data = list()
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            line_data = json.loads(line.strip())
            data.append(line_data)
    return data


def load_json(filepath, **kwargs):
    with open(filepath, "rt", encoding="utf-8") as fin:
        data = json.load(fin, **kwargs)
    return data


def dump_json(obj, filepath, **kwargs):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wt", encoding="utf-8") as fout:
        json.dump(obj, fout, indent=4, ensure_ascii=False, **kwargs)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
