from transformers import AutoTokenizer
import torch
import argparse
import re
import os
import torch
from transformers import AutoTokenizer

from trans.dataset import InferenceBTDataset, InferenceBTConstrainDataset
from trans.utils.io import dump_jsonlines, load_jsonlines
from vllm import LLM, SamplingParams

class BTInfer:
    def __init__(self, model_path, tensor_parallel_size):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = LLM(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
        )

    @torch.inference_mode()
    def infer(self, 
              data_path,
              save_dir,
              source,
              target,
              temperature,
              top_p,
              max_new_tokens,
              infer_cnt):

        source = source.upper()
        target = target.upper()
        
        def _parse_output(text):
            pattern = fr"\[{target}\](.*?)\[/{target}\]"
            match = re.search(pattern, text, re.DOTALL)
            text_content = ""
            if match:
                text_content = match.group(1).strip()
            return text_content
        
        if infer_cnt == 0:
            dataset = InferenceBTConstrainDataset(
                load_jsonlines(data_path),
                self.tokenizer,
                source,
                target
            )
        else:
            dataset = InferenceBTDataset(
                load_jsonlines(data_path),
                self.tokenizer,
                source,
                target
            )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        prompts = dataset.get_all()

        # 00:25 / 100 prompts on one GPU
        results = self.llm.generate(prompts, use_tqdm=True, sampling_params=sampling_params)

        raw = dataset.get_raw()

        dump_jsonl = []
        for raw, result in zip(raw, results):
            raw_output = result.outputs[0].text
            output = raw_output

            if infer_cnt == 0:
                output = _parse_output(raw_output)
                if output == "":
                    continue

            dump_jsonl.append(
                {
                    source: raw[source],
                    f'{target}-gen': output,
                }
            )
        save_filepath = os.path.join(save_dir, f"{source}-{target}-gen_{infer_cnt}.jsonl")
        dump_jsonlines(dump_jsonl, save_filepath)
        print(f'saved to {save_filepath}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--source", type=str, default="cpp")
    parser.add_argument("--target", type=str, default="cuda")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--infer_cnt", type=int, default=0)
    args = parser.parse_args()

    infer = BTInfer(args.model_path, args.tensor_parallel_size)
    infer.infer(
        args.data_path,
        args.save_dir,
        args.source,
        args.target,
        args.temperature,
        args.top_p,
        args.max_new_tokens,
        args.infer_cnt
    )