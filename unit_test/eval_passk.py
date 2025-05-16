import itertools
import json
import evaluate
from datetime import datetime
from traceback import format_exc
from typing import List, Dict, Union
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Tuple
from datetime import datetime
import threading

from traceback import format_exc

from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd
from unit_test.compiler import CppCompiler, CompileMethod, CudaCompiler, CompilingResult
from unit_test.configs import RESOURCES_DIR, OUTPUT_DIR, EVAL_TRANSLATED_OUTPUT_DIR
from unit_test.schemas import UnitTestEvalCase, UnitTestEvalResultModel
from unit_test.utils.common_utils import remove_code_block_lines, get_function_name_from_cpp_or_cuda_code, \
    replace_kernel_function_in_wrapper, replace_wrapper_func_first_arg
from unit_test.utils.json_utils import load_jsonl_file

from models import ModelFactory, InferModel, PromptType
from trans.dataset import TransDirection
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def eval_bleu(pre, ref):
    bleu_metric = evaluate.load("trans/evaluation/bleu")
    codebleu_metric = evaluate.load("trans/evaluation/codebleu")
    bleu_metric.add_batch(predictions=pre, references=ref)
    codebleu_metric.add_batch(predictions=pre, references=ref)
    return bleu_metric.compute()['bleu'], codebleu_metric.compute(lang='cpp', weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)['codebleu']


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def check(result: dict) -> bool:
    if result["cpp_result"] == 'success' and result["cuda_result"] == 'success':
        cpp_outputs = result["cpp_output"].strip().split("\n")
        cuda_outputs = result["cuda_output"].strip().split("\n")
        if len(cpp_outputs) == len(cuda_outputs):
            normalized_cpp = list(
                map(CodeEvaluator._normalize_output, cpp_outputs))
            normalized_cuda = list(
                map(CodeEvaluator._normalize_output, cuda_outputs))
            matches = sum(1 for c, d in zip(
                normalized_cpp, normalized_cuda) if c == d)
            if matches == len(cpp_outputs):
                return True
    return False


def check_compile_helper(result: dict) -> bool:
    if result["cuda_result"] != CompilingResult.COMPILE_ERROR:
        return True


def check_compile(results: List[dict]) -> bool:
    for result in results:
        if not check_compile_helper(result):
            return False
    return True


def check_match(results: List[dict]) -> bool:
    for result in results:
        if not check(result):
            return False
    return True


class CodeEvaluator:
    def __init__(self):
        self.cpp_compiler = CppCompiler(CompileMethod.LOCAL)
        self.cuda_compiler = CudaCompiler(CompileMethod.LOCAL)

    def run_test_cases(self, test_cases: List[UnitTestEvalCase]) -> List[Dict]:
        chunksize = len(test_cases) // 32
        if chunksize == 0:
            chunksize = 1
        logger.info(
            f"Processing {len(test_cases)} test cases in parallel with chunksize {chunksize}")
        results = []
        with Pool() as pool:
            results = list(tqdm(
                pool.imap_unordered(self._run_test_case,
                                    test_cases, chunksize=chunksize),
                total=len(test_cases),
                desc="Evaluating functions",
                mininterval=0.1,  # Update at least every 0.1 seconds
                maxinterval=1.0   # Update at most every 1.0 seconds
            ))
        return results

    def _run_test_case(self, test_case: UnitTestEvalCase) -> List[Dict]:
        results = []

        for cpp_code, cuda_code in zip(test_case.format_cpp_code(), test_case.format_cuda_code()):
            # Run CPP compilation
            cpp_result, cpp_output, cpp_meta_data = self.cpp_compiler.run_code(
                cpp_code)
            # Run CUDA compilation
            cuda_result, cuda_output, cuda_meta_data = self.cuda_compiler.run_code(
                cuda_code)

            results.append({
                "cpp_result": cpp_result,
                "cuda_result": cuda_result,
                "cpp_output": cpp_output,
                "cuda_output": cuda_output,
                "cpp_compiling_time": cpp_meta_data['compile_time'],
                "cpp_execution_time": cpp_meta_data['execution_time'],
                "cuda_compiling_time": cuda_meta_data['compile_time'],
                "cuda_execution_time": cuda_meta_data['execution_time'],
            })

        return results

    @staticmethod
    def _normalize_output(output: str) -> str:
        """Normalize output string for comparison"""
        return ' '.join(output.strip().split())


# Create thread lock for counter synchronization
counter_lock = threading.Lock()


def translate(
    model: InferModel, test_cases: List[UnitTestEvalCase], direction: TransDirection, sample_num: int
) -> List[str]:
    inputs = []
    systems = []
    for test_case in test_cases:
        if direction.source == "CPP":
            code = test_case.cpp_code
        else:
            code = test_case.cuda_code
        system, input = model.generate_prompt(code, direction)
        systems.append(system)
        inputs.append(input)
    # save the prompts here,  
    # save_jsonl_path = RESOURCES_DIR / "eval_trans_prompts.jsonl"
    # with open(save_jsonl_path, "w", encoding="utf-8") as f:
    #     for system, input in zip(systems, inputs):
    #         messages = [
    #             # {"role": "system", "content": system},
    #             {"role": "user", "content": input},
    #         ]
    #         f.write(json.dumps(messages) + "\n")
    return model.collect_batch(systems, inputs, sample_num)


def process_single_test_case(
        idx: int, target: str, test_case: UnitTestEvalCase, log_data: dict, direction: TransDirection
) -> Tuple[int, bool, dict]:
    """
    Process single test case with retry mechanism
    Returns: (case_id, is_success, case_log)
    """

    test_case_log = {
        "index": idx,
        "cpp_code": test_case.cpp_code,
        "original_cuda_code": test_case.cuda_code,
        "cuda_wrapper": test_case.cuda_wrapper,
        "translation_success": False,
        "translated_cuda_code": None,
        "error": None
    }
    test_case.source = direction.source
    try:
        if direction.source == "CPP":
            # target: CUDA kernel
            generated_cuda_kernel_function_name = get_function_name_from_cpp_or_cuda_code(
                target)
            __wrapper = replace_kernel_function_in_wrapper(
                test_case.cuda_wrapper, generated_cuda_kernel_function_name)
        else:
            __wrapper = test_case.cuda_wrapper
        test_case_log["translation_success"] = True
        test_case_log["translated_cuda_code"] = target
        test_case_log["modified_wrapper"] = __wrapper
        idx, is_success, case_log, translation_result = idx, True, test_case_log, (
            target, __wrapper)

    except Exception as e:
        test_case_log["error"] = str(e)
        logger.error(format_exc())
        idx, is_success, case_log, translation_result = idx, True, test_case_log, (
            "", "")

    log_data["test_cases"].append(case_log)
    if is_success and translation_result:
        if direction.source == "CPP":
            # target: CUDA kernel
            cuda_code, wrapper = translation_result
            test_case.cuda_code = cuda_code
            test_case.cuda_wrapper = wrapper
        else:
            # target: CPP
            cpp_code, wrapper = translation_result
            test_case.cpp_code = cpp_code
            test_case.cuda_wrapper = wrapper
            # replace function_name consistent inputs for C++ test cases
            test_case.consistent_cpp_inputs = [replace_wrapper_func_first_arg(
                cpp_input, get_function_name_from_cpp_or_cuda_code(cpp_code)) for cpp_input in test_case.consistent_cpp_inputs]


def evaluate_llm(model: str, test_cases: List[UnitTestEvalCase], source: str, file_path: str = None):
    # Create logs directory if it doesn't exist
    logs_dir = OUTPUT_DIR / "eval_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for log filename
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    target = "cpp" if args.source == "cuda" else "cuda"
    log_data = {
        "model": model,
        "timestamp": timestamp,
        "test_cases": [],
        "pass@5": 0,
        "pass@10": 0,
    }
    if model.endswith("/"):
        model_name = model[:-1]
    else:
        model_name = model
    model_name = model_name.replace("/", "_")

    log_filename = f"{args.source}2{target}_{model_name}_{timestamp}"
    log_path = logs_dir / log_filename

    logger.info(f"test_cases size:{len(test_cases)}")

    direction = TransDirection(source=source)

    n_sample = args.sample_num

    targets = []
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                target_code = line.strip()
                targets.append(target_code)
    else:
        model = ModelFactory.get_model(model, mode=args.model_mode)
        # Translate Cpp to Cuda
        # from loguru import logger
        logger.debug(f"test_cases size: {len(test_cases)}")
        logger.debug(f"model: {model}")
        targets = translate(model, test_cases, direction, n_sample)
        assert len(targets) == (len(test_cases) * n_sample)

        if args.save_translated_code:
            save_path = EVAL_TRANSLATED_OUTPUT_DIR / \
                f"{args.source}2{target}_{model_name}_pass@k.txt"
            with open(save_path, "w", encoding="utf-8") as f:
                for target in targets:
                    target = target.replace('\n', ' ').strip()
                    f.write(target + "\n")

    # Duplicate each test case n_sample times
    # TODO: duplicate execute of source code
    test_cases = [
        test_case for test_case in test_cases for _ in range(n_sample)]
    # print(f"test_cases size: {len(test_cases)}")
    # print(f"targets size: {len(targets)}")
    assert len(test_cases) == len(targets)

    targets_first = targets[::n_sample]
    reference_codes = [
        test_case.cuda_code for test_case in test_cases[::n_sample]]
    # logger.debug(f"targets_first: {targets_first}")
    # logger.debug(f"reference_codes: {reference_codes}")
    bleu, code_bleu = eval_bleu(targets_first, reference_codes)

    logger.info(f"BLEU score: {bleu}")
    logger.info(f"CodeBLEU score: {code_bleu}")
    # replace funcion_name in wrapper, from now, test_case.cuda is targets
    for idx, (target, test_case) in enumerate(zip(targets, test_cases)):
        process_single_test_case(
            idx, target, test_case, log_data, direction=direction)

    evaluator = CodeEvaluator()
    # Process test cases in parallel with progress bar
    results = evaluator.run_test_cases(test_cases)

    assert len(results) == len(test_cases)
    success = []
    compile_succes = []
    for function_results, test_case in zip(results, test_cases):
        if check_match(function_results):
            success.append(1)
        else:
            success.append(0)
        if check_compile(function_results):
            compile_succes.append(1)
        else:
            compile_succes.append(0)
    # Take one element for every n_sample elements to use for BLEU calculation
    # Group success results by n_sample
    grouped_success = np.array(success).reshape(-1, n_sample).sum(axis=1)
    pass_1 = estimate_pass_at_k(n_sample, grouped_success, 1)
    pass_5 = estimate_pass_at_k(n_sample, grouped_success, 5)
    pass_10 = estimate_pass_at_k(n_sample, grouped_success, 10)

    grouped_compile_success = np.array(
        compile_succes).reshape(-1, n_sample).sum(axis=1)
    compile_pass_1 = estimate_pass_at_k(n_sample, grouped_compile_success, 1)

    # Save pass@5 and pass@10 scores to an Excel file
    excel_data = {
        "Pass@1": pass_1,
        "Pass@5": pass_5,
        "Pass@10": pass_10
    }
    df = pd.DataFrame(excel_data)
    excel_path = OUTPUT_DIR / f"{log_filename}_pass@k.xlsx"
    df.to_excel(excel_path, index=False)
    logger.info(f"Pass@1, Pass@5 and Pass@10 scores saved to: {excel_path}")
    logger.info(f"Compile pass@1 score: {compile_pass_1.mean()}")
    logger.info(f"Pass@1 score: {pass_1.mean()}")
    logger.info(f"Pass@5 score: {pass_5.mean()}")
    logger.info(f"Pass@10 score: {pass_10.mean()}")

    log_data["bleu"] = bleu
    log_data["code_bleu"] = code_bleu
    log_data["cpass"] = compile_pass_1.mean()
    log_data["pass@1"] = pass_1.mean()
    log_data["pass@5"] = pass_5.mean()
    log_data["pass@10"] = pass_10.mean()
    # Save log data to JSON file
    with open(f"{log_path}.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation logs saved to: {log_path}.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_eval", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--source", type=str, default='cpp')
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--model_mode", type=PromptType,
                        default=PromptType.TRANS_TRAINED)
    parser.add_argument("--sample_num", type=int, default=20)
    parser.add_argument("--save_translated_code", action="store_true")
    args = parser.parse_args()
    model_to_eval = args.model_to_eval
    eval_unittest_cases = load_jsonl_file(
        RESOURCES_DIR / "unit_total_eval_cases.jsonl", pydantic_model=UnitTestEvalCase)
    eval_unittest_cases = eval_unittest_cases
    evaluate_llm(model=model_to_eval, test_cases=eval_unittest_cases,
                 source=args.source, file_path=args.file_path)
