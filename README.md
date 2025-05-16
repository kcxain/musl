## Mutual-Supervised Learning for Sequential-to-Parallel Code Translation

## Install
```bash
pip install -e .
```

## Structure

- original/filtered data: `BabelTower/dataset`
- test set with unit test: `resources/unit_total_eval_cases.jsonl`
- a unified framework for inference `models/base`
- codebase for co-verify `unit_test`
- codebase for co-evolve `trans`

## Usage

### Co-verify

```bash
bash scripts/build_sft.sh
```
This script will use vllm to inference and apply co-verify to build the sft data for code translation and unit test generation.

### Co-Evolve
We use llama-factory for fine-tuning.
```bash
git clone https://github.com/hiyouga/LLaMA-Factory
```
You can register the dataset from Co-verify and fine-tune the model according to the llama-factory docs.

### Evaluate Pass@k

```bash
bash scripts/eval_pass_k.sh
```