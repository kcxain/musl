from transformers import AutoTokenizer
from transformers import AutoTokenizer

from trans.utils.prompts import CUDA_CPP_TRANSLATE_TRAIN_SYSTEM
from trans.dataset import TransDirection
from trans.utils.io import dump_jsonlines, load_jsonlines
from vllm import LLM, SamplingParams
import argparse
def predict(model_path, tensor_parallel_size, data_path, output_path):
    sampling_params = SamplingParams(
        max_tokens=2048,
        n=1,
    )
    llm = LLM(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = []

    data = load_jsonlines(data_path)
    if 'CUDA' in data[0]:
        direction = TransDirection('cuda', 'cpp')
    elif 'CPP' in data[0]:
        direction = TransDirection('cpp', 'cuda')
    else:
        raise ValueError("Invalid data format")
    
    for d in data:
        message = [
            {"role": "system","content": CUDA_CPP_TRANSLATE_TRAIN_SYSTEM.format(obj=direction)},
            {"role": "user", "content": d[direction.source]},
        ]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    
    results = llm.generate(prompts, use_tqdm=True, sampling_params=sampling_params)
    
    validate_json = []
    usft_json = []
    for result, raw_data in zip(results, data):
        for output in result.outputs:
            validate_json.append({
                direction.source: raw_data[direction.source],
                direction.target: output.text,
                "source": direction.source,
            })

    dump_jsonlines(validate_json, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tensor_parallel_size', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    predict(args.model_path, args.tensor_parallel_size, args.data_path, args.output_path)