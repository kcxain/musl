from datetime import datetime
from html import parser
from typing import List
from copy import deepcopy
from pathlib import Path
from unit_test.configs import RESOURCES_DIR, TRANSLATION_VALIDATION_OUTPUT_DIR, QWEN_MODEL_PATH
from unit_test.generator import UnitTestGenerator
from unit_test.eval import CodeEvaluator
from unit_test.schemas import TranslationInput, ValidTranslation
from tqdm import tqdm
import orjson as json
from multiprocessing import Pool
from models import PromptType


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


class TranslationValidator(object):
    def __init__(self, translation_input: TranslationInput, round: str = None):
        self.translation_input = translation_input
        self.translation_input.load_from_jsonl()
        format_time = datetime.now().strftime("%m-%d-%H-%M")
        self.project_output_dir = TRANSLATION_VALIDATION_OUTPUT_DIR / \
            "qwen3" / f"round{round}-{format_time}"
        self.project_output_dir.mkdir(exist_ok=True, parents=True)

    def check_match(self, results: List[dict]) -> bool:
        for result in results:
            if not check(result):
                return False
        return True

    def validate_translations(self, save: bool = True, debug: bool = False) -> List[ValidTranslation]:
        generator = UnitTestGenerator(
            model=self.translation_input.model_path, mode=self.translation_input.mode)

        valid_translations = []

        pairs = self.translation_input.translation_pairs
        if debug:
            pairs = pairs[:100]

        eval_cases = generator.generate_parallel_unit_tests(
            pairs, output_dir=self.project_output_dir)

        evaluator = CodeEvaluator()
        # Process test cases in parallel with progress bar
        results = evaluator.run_test_cases(eval_cases)

        valid_translations = []
        assert len(results) == len(eval_cases)
        for function_results, eval_case in zip(results, eval_cases):
            if debug:
                logger.info(eval_case.cuda_code)
                logger.info(eval_case.cpp_code)
                logger.info(eval_case.cuda_wrapper)
                logger.info(eval_case.consistent_cpp_inputs)
                logger.info(eval_case.consistent_cuda_inputs)
                logger.info(function_results)
            if self.check_match(function_results):
                valid_unit_tests_inputs = eval_case.consistent_cpp_inputs if eval_case.source == 'CPP' else eval_case.consistent_cuda_inputs
                if valid_unit_tests_inputs:
                    valid_translations.append(ValidTranslation(
                        CUDA=eval_case.cuda_code,
                        CPP=eval_case.cpp_code,
                        CUDA_WRAPPER=eval_case.cuda_wrapper,
                        source=eval_case.source,
                        valid_unit_tests_inputs=valid_unit_tests_inputs
                    ))
        # print(results)
        if save:
            output_file = self.project_output_dir / \
                f"source_{eval_case.source}.jsonl"
            with open(output_file, 'wb') as f:
                for translation in valid_translations:
                    f.write(json.dumps(translation.model_dump()))
                    f.write(b'\n')

        return valid_translations


if __name__ == "__main__":
    from loguru import logger
    import argparse
    parser = argparse.ArgumentParser(description="Translation Validator")
    parser.add_argument("--cpp_file", type=str, default=None,
                        help="Path to the cpp input file")
    parser.add_argument("--cuda_file", type=str, default=None,
                        help="Path to the cuda input file")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict check")
    parser.add_argument("--test_prompt_mode", type=str, default="test_trained")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--round", type=str)
    args = parser.parse_args()
    cpp_input = Path(args.cpp_file)
    cuda_input = Path(args.cuda_file)
    translation_input = TranslationInput(model_path=args.model_path,
                                         input_file=cpp_input.as_posix(),
                                         mode=args.test_prompt_mode,)
    translation_validator = TranslationValidator(
        translation_input, round=args.round)
    valida_res = translation_validator.validate_translations()

    translation_input = TranslationInput(model_path=args.model_path,
                                         input_file=cuda_input.as_posix(),
                                         mode=args.test_prompt_mode)
    translation_validator = TranslationValidator(
        translation_input, round=args.round)
    valida_res = translation_validator.validate_translations()
