import json
from datetime import datetime
from traceback import format_exc
from typing import List, Dict
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Tuple
from datetime import datetime
import threading

from traceback import format_exc

from loguru import logger
from tqdm import tqdm
import numpy as np
import evaluate

from unit_test.compiler import CppCompiler, CompileMethod, CudaCompiler, CompilingResult
from unit_test.configs import RESOURCES_DIR, OUTPUT_DIR, EVAL_TRANSLATED_OUTPUT_DIR
from unit_test.schemas import UnitTestEvalCase, UnitTestEvalResultModel
from unit_test.utils.common_utils import remove_code_block_lines, get_function_name_from_cpp_or_cuda_code, \
    replace_kernel_function_in_wrapper, replace_wrapper_func_first_arg
from unit_test.utils.json_utils import load_jsonl_file

from models import ModelFactory, InferModel, PromptType
from trans.dataset import TransDirection


class CodeEvaluator:
    def __init__(self):
        self.cpp_compiler = CppCompiler(CompileMethod.LOCAL)
        self.cuda_compiler = CudaCompiler(CompileMethod.LOCAL)

    def run_test_cases(self, test_cases: List[UnitTestEvalCase]) -> List[Dict]:
        chunksize = len(test_cases) // 32
        if chunksize == 0:
            chunksize = 1
        logger.info(f"Processing {len(test_cases)} test cases in parallel with chunksize {chunksize}")
        results = []
        with Pool() as pool:
            results = list(tqdm(
                pool.imap(self._run_test_case, test_cases, chunksize=chunksize),
                total=len(test_cases),
                desc="Evaluating functions"
            ))
        return results
    
    def _run_test_case(self, test_case: UnitTestEvalCase) -> List[Dict]:
        results = []

        for cpp_code, cuda_code in zip(test_case.format_cpp_code(), test_case.format_cuda_code()):
            # Run CPP compilation
            cpp_result, cpp_output, cpp_meta_data = self.cpp_compiler.run_code(cpp_code)
            # Run CUDA compilation
            cuda_result, cuda_output, cuda_meta_data = self.cuda_compiler.run_code(cuda_code)

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

    def evaluate(self, test_cases: List[UnitTestEvalCase], log_data: dict) -> UnitTestEvalResultModel:
        total_functions = len(test_cases)
        total_cases = sum(len(case.format_cpp_code()) for case in test_cases)

        # Initialize metrics
        results = self.run_test_cases(test_cases)

        # Process test cases in parallel with progress bar
        # for case in tqdm(test_cases):
        #     results.append(self._run_test_case(case))

        # Initialize counters
        cpp_compile_success = 0
        cuda_compile_success = 0
        cpp_run_success = 0
        cuda_run_success = 0
        matching_cases = 0
        matching_functions = 0
        speedup_ratios = []

        # Process results
        for _idx, function_results in enumerate(results):
            log_data["test_cases"][_idx]["function_results"] = function_results
            function_cases_match = True
            for result in function_results:
                # Count successful compilations
                if result["cpp_result"] != CompilingResult.COMPILE_ERROR:
                    cpp_compile_success += 1
                if result["cuda_result"] != CompilingResult.COMPILE_ERROR:
                    cuda_compile_success += 1

                # Count successful runs
                if result["cpp_result"] == CompilingResult.SUCCESS:
                    cpp_run_success += 1
                if result["cuda_result"] == CompilingResult.SUCCESS:
                    cuda_run_success += 1

                # Check output matching
                if (result["cpp_result"] == CompilingResult.SUCCESS and
                        result["cuda_result"] == CompilingResult.SUCCESS):
                    cpp_outputs = result["cpp_output"].strip().split("\n")
                    cuda_outputs = result["cuda_output"].strip().split("\n")
                    if len(cpp_outputs) == len(cuda_outputs):
                        # Parallel output normalization and comparison
                        # with Pool() as pool:
                        #     normalized_cpp = pool.map(self._normalize_output, cpp_outputs)
                        #     normalized_cuda = pool.map(self._normalize_output, cuda_outputs)
                        normalized_cpp = list(map(self._normalize_output, cpp_outputs))
                        normalized_cuda = list(map(self._normalize_output, cuda_outputs))
                        matches = sum(1 for c, d in zip(normalized_cpp, normalized_cuda) if c == d)

                        if matches == len(cpp_outputs):
                            matching_cases += 1
                            if result["cpp_execution_time"] > 0:
                                speedup = result["cpp_execution_time"] / result["cuda_execution_time"]
                                speedup_ratios.append(speedup)
                        else:
                            function_cases_match = False
                    else:
                        function_cases_match = False
                else:
                    function_cases_match = False

            if function_cases_match:
                matching_functions += 1

        # Calculate final metrics
        if test_cases[0].source == 'CPP':
            compiling_pass_rate = cuda_compile_success / total_cases
            running_pass_rate = cuda_run_success / total_cases
        else:
            compiling_pass_rate = cpp_compile_success / total_cases
            running_pass_rate = cpp_run_success / total_cases
        matching_rate_cases = matching_cases / total_cases if total_cases > 0 else 0
        matching_rate_functions = matching_functions / total_functions if total_functions > 0 else 0
        average_speedup = np.mean(speedup_ratios) if speedup_ratios else 1.0

        return UnitTestEvalResultModel(
            model_to_eval=args.model_to_eval,
            total_functions=total_functions,
            total_cases=total_cases,
            compiling_pass_rate=compiling_pass_rate,
            running_pass_rate=running_pass_rate,
            matching_rate_cases=matching_rate_cases,
            matching_rate_functions=matching_rate_functions,
            average_speedup_ratio=average_speedup
        )

    @staticmethod
    def _normalize_output(output: str) -> str:
        """Normalize output string for comparison"""
        return ' '.join(output.strip().split())


def evaluate_llm_generated_code(test_cases: List[UnitTestEvalCase], log_data: dict) -> UnitTestEvalResultModel:
    evaluator = CodeEvaluator()
    return evaluator.evaluate(test_cases, log_data)


# Create thread lock for counter synchronization
counter_lock = threading.Lock()

def translate(
    model: InferModel, test_cases: List[UnitTestEvalCase], direction: TransDirection
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
    return model.collect_batch(systems, inputs)
    

def eval_bleu(pre, ref):
    bleu_metric = evaluate.load("trans/evaluation/bleu")
    codebleu_metric = evaluate.load("trans/evaluation/codebleu")
    bleu_metric.add_batch(predictions=pre, references=ref)
    codebleu_metric.add_batch(predictions=pre, references=ref)
    logger.info(f"BLEU: {bleu_metric.compute()['bleu']}")
    logger.info(f"CodeBLEU: {codebleu_metric.compute(lang='cpp', weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)['codebleu']}")


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
            generated_cuda_kernel_function_name = get_function_name_from_cpp_or_cuda_code(target)
            __wrapper = replace_kernel_function_in_wrapper(test_case.cuda_wrapper, generated_cuda_kernel_function_name)
        else:
            __wrapper = test_case.cuda_wrapper
        test_case_log["translation_success"] = True
        test_case_log["translated_cuda_code"] = target
        test_case_log["modified_wrapper"] = __wrapper
        idx, is_success, case_log, translation_result = idx, True, test_case_log, (target, __wrapper)

    except Exception as e:
        test_case_log["error"] = str(e)
        logger.error(format_exc())
        idx, is_success, case_log, translation_result = idx, True, test_case_log, ("", "")
    
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
            test_case.consistent_cpp_inputs = [replace_wrapper_func_first_arg(cpp_input, get_function_name_from_cpp_or_cuda_code(cpp_code)) for cpp_input in test_case.consistent_cpp_inputs]


def evaluate_llm(model: str, test_cases: List[UnitTestEvalCase], source: str, file_path: str = None):
    # Create logs directory if it doesn't exist
    logs_dir = OUTPUT_DIR / "eval_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for log filename
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    target = "cpp" if args.source == "cuda" else "cuda"
    log_filename = f"{args.source}2{target}_{model.replace('/', '_')}_{timestamp}"
    log_path = logs_dir / log_filename

    # Initialize log data structure
    log_data = {
        "model": model,
        "timestamp": timestamp,
        "test_cases": [],
        "evaluation_results": None
    }

    logger.info(f"test_cases size:{len(test_cases)}")

    direction = TransDirection(source=source)
    targets = []
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                target_code = line.strip()
                targets.append(target_code)
    else:
        model = ModelFactory.get_model(model, mode=args.model_mode)
        # Translate Cpp to Cuda
        targets = translate(model, test_cases, direction)
        if args.save_translated_code:
            # {source}_{model}.txt
            save_path = EVAL_TRANSLATED_OUTPUT_DIR / f"{args.source}_{args.model_to_eval}.txt"
            with open(save_path, "w", encoding="utf-8") as f:
                for target in targets:
                    target = target.replace('\n', ' ').strip()
                    f.write(target + "\n")

    for target in targets:
        print(target)
    # evaluate BLEU
    if target == 'cpp':
        eval_bleu(targets, [test_case.cpp_code for test_case in test_cases])
    else:
        eval_bleu(targets, [test_case.cuda_code for test_case in test_cases])
    
    # replace funcion_name in wrapper
    for idx, (target, test_case) in enumerate(zip(targets, test_cases)):
        process_single_test_case(idx, target, test_case, log_data, direction=direction)

    # print(targets)
    # Evaluate translated code
    eval_res = evaluate_llm_generated_code(test_cases, log_data)
    log_data["evaluation_results"] = eval_res.model_dump()
    log_data["eval_results"] = eval_res.model_dump()
    # Save log data to JSON file
    with open(f"{log_path}.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # Print results
    from pprint import pprint
    pprint(eval_res)
    logger.info(f"Evaluation logs saved to: {log_path}.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_eval", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--source", type=str, default='cpp')
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--model_mode", type=PromptType, default=PromptType.TRANS_TRAINED)
    parser.add_argument("--save_translated_code", action="store_true")
    args = parser.parse_args()
    model_to_eval = args.model_to_eval
    eval_unittest_cases = load_jsonl_file(RESOURCES_DIR / "unit_total_eval_cases.jsonl", pydantic_model=UnitTestEvalCase)
    eval_unittest_cases = eval_unittest_cases
    evaluate_llm(model=model_to_eval, test_cases=eval_unittest_cases, source=args.source, file_path=args.file_path)