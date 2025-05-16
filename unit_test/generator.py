import argparse
from pathlib import Path
from typing import List

from unit_test.configs import ROOT, CUDA_WRAPPER_MODEL_PATH, CUDA_WRAPPER_MODEL_PATH_TRAINED

from unit_test.schemas import CppUnitTestModel, CudaUnitTestModel, UnitTestModel, TranslationPair, UnitTestEvalCase
from unit_test.utils.common_utils import (
    get_function_name_from_cpp_or_cuda_code,
    replace_assert_statements,
    wrapper_function_invoke_to_print_variables,
    load_tok_file,
    replace_kernel_function_in_wrapper,
    replace_wrapper_func_first_arg,
    remove_comments
)

from trans.dataset import TransDirection
from models import InferModel, ModelFactory, PromptType
from loguru import logger
from trans.utils.io import rank0_print
import concurrent.futures
import copy

class UnitTestGenerator:
    def __init__(self, model: str, mode: PromptType):
        # lazy load model
        self.test_model_name = model
        self.mode = mode

    def generate_cuda_wrapper(self, cuda_codes: List[str], cuda_wrapper_model_path=CUDA_WRAPPER_MODEL_PATH_TRAINED) -> List[str]:
        # cuda_wrapper_model = ModelFactory.get_model(cuda_wrapper_model_path, mode='cuda_wrapper')
        cuda_wrapper_model = ModelFactory.get_model(CUDA_WRAPPER_MODEL_PATH_TRAINED, mode='cuda_wrapper_trained')
        wrapper_systems, wrapper_prompts = [], []
        unique_cuda_codes = list(set(cuda_codes))
        cuda_code_to_index = {code: idx for idx, code in enumerate(unique_cuda_codes)}
        
        for cuda_code in unique_cuda_codes:
            wrapper_system, wrapper_prompt = cuda_wrapper_model.generate_prompt(cuda_code, None)
            wrapper_systems.append(wrapper_system)
            wrapper_prompts.append(wrapper_prompt)
        
        unique_wrapper_cudas = cuda_wrapper_model.collect_batch(systems=wrapper_systems, inputs=wrapper_prompts)
        wrapper_cudas = [unique_wrapper_cudas[cuda_code_to_index[cuda_code]] for cuda_code in cuda_codes]
        # release wrapper model
        del cuda_wrapper_model
        # NOTE: to avoid generate wrong function name
        for i in range(len(wrapper_cudas)):
            wrapper_cudas[i] = replace_kernel_function_in_wrapper(wrapper_cudas[i], get_function_name_from_cpp_or_cuda_code(cuda_codes[i]))
        return wrapper_cudas
    
    def generate_unit_tests(self, function_codes: List[str], directions: List[TransDirection], cached_cuda_wrapper: List[str] = None, cuda_wrapper_model_path=CUDA_WRAPPER_MODEL_PATH_TRAINED) -> List[UnitTestModel]:
        """
        with the given function codes, generate the unit tests. 
        directions[i].source indicates the language of function_codes[i].
        NOTE: the List of function codes may not be the same language, so we need to specify the source code of each function code.
        """
        # there is no need to train cuda_wrapper_model，we always use Llama-3-8B-Instruct
        # first, generate the wrapper code for the given function codes
        cuda_codes = [function_code for function_code, direction in zip(function_codes, directions) if direction.source == "CUDA"]
        if cuda_codes:
            # generate_parallel_unit_tests has already generated the wrapper
            if cached_cuda_wrapper and len(cached_cuda_wrapper) == len(cuda_codes):
                wrapper_cudas = copy.deepcopy(cached_cuda_wrapper)
            else:
                wrapper_cudas = self.generate_cuda_wrapper(cuda_codes, cuda_wrapper_model_path)
        # rank0_print(wrapper_cudas)
        logger.info("Loading Unit Test Model...")
        self.model = ModelFactory.get_model(self.test_model_name, mode=self.mode)
        
        function_names, systems, prompts = [], [], []
        wrapper_function_names, kernel_function_names = [], []
        cuda_coda_idx = 0
        for code, direction in zip(function_codes, directions):
            if direction.source == "CPP":
                function_names.append(get_function_name_from_cpp_or_cuda_code(code))
            elif direction.source == "CUDA":
                # if the code is CUDA, function name is wrapper's
                wrapper_function_name = get_function_name_from_cpp_or_cuda_code(wrapper_cudas[cuda_coda_idx])
                wrapper_function_names.append(wrapper_function_name)
                kernel_function_names.append(get_function_name_from_cpp_or_cuda_code(code))
                function_names.append(wrapper_function_name)
                code = code + '\n' + wrapper_cudas[cuda_coda_idx]
                cuda_coda_idx += 1
            else:
                raise ValueError("Invalid source code")
            system, prompt = self.model.generate_prompt(code, direction)
            systems.append(system)
            prompts.append(prompt)

        unique_systems_prompts = list(set(zip(systems, prompts)))
        unique_systems, unique_prompts = zip(*unique_systems_prompts)
        test_cases_list = self.model.collect_batch(systems=list(unique_systems), inputs=list(unique_prompts))
        
        # Map the results back to the original order
        system_prompt_to_test_cases = {sp: tc for sp, tc in zip(unique_systems_prompts, test_cases_list)}
        test_cases_list = [system_prompt_to_test_cases[(system, prompt)] for system, prompt in zip(systems, prompts)]
        
        del self.model
        size = len(test_cases_list)
        replaced_test_cases_list = [[" "]] * size
        for i in range(len(test_cases_list)):
            parsed_cases = []
            for case in test_cases_list[i]:
                try:
                    parsed_case = wrapper_function_invoke_to_print_variables(
                            replace_assert_statements(remove_comments(case)), function_names[i]
                        )
                except Exception as e:
                    logger.error(f"When processing item {i}, Error: {e}")
                    logger.error(f"Original test case: {test_cases_list[i]}")
                    parsed_case = " "
                    # rank0_print(test_cases_list[i])
                parsed_cases.append(parsed_case)
            replaced_test_cases_list[i] = parsed_cases
            
        res = []
        for function_code, parsed_output, direction in zip(function_codes, replaced_test_cases_list, directions):
            if direction.source == "CPP":
                res.append(CppUnitTestModel(
                    function_code=function_code,
                    test_cases=parsed_output,
                ))

            elif direction.source == "CUDA":
                res.append(CudaUnitTestModel(
                    kernel_code_function_name = kernel_function_names.pop(0),
                    kernel_code=function_code,
                    wrapper_function=wrapper_cudas.pop(0),
                    wrapper_function_function_name=wrapper_function_names.pop(0),
                    test_cases=parsed_output,
                ))
            
            else:
                raise ValueError("Invalid source code")
        return res

    def _replaces_wrapper_function_name(self, test_cases: List[str], function_name: str) -> List[str]:
        return [replace_wrapper_func_first_arg(case, function_name) for case in test_cases]

    def generate_parallel_unit_tests(self, translation_pairs: List[TranslationPair], cuda_wrapper_model_path=CUDA_WRAPPER_MODEL_PATH_TRAINED, output_dir: str = "unit_tests") -> List[UnitTestEvalCase]:
        function_codes = {
            "CPP": [],
            "CPP_FUNCTION_NAME": [],
            "CUDA": [],
            "CUDA_KERNEL_FUNCTION_NAME": [],
            "CUDA_WRAPPER": [],
            "CUDA_WRAPPER_FUNCTION_NAME": [],
            "TEST_CASES": [],
            "DIRECTION": [],
            "CODE_TO_TEST": [],
            "TEST_CASES": []
        }
        # prepare function codes
        for pair in translation_pairs:
            function_codes["CPP"].append(pair.CPP)
            function_codes["CUDA"].append(pair.CUDA)
            function_codes["DIRECTION"].append(pair.direction)
            function_codes["CPP_FUNCTION_NAME"].append(get_function_name_from_cpp_or_cuda_code(pair.CPP))
            function_codes["CUDA_KERNEL_FUNCTION_NAME"].append(get_function_name_from_cpp_or_cuda_code(pair.CUDA))
        # generate wrapper
        logger.info("Generating CUDA wrapper...")
        function_codes["CUDA_WRAPPER"] = self.generate_cuda_wrapper(function_codes["CUDA"],cuda_wrapper_model_path=cuda_wrapper_model_path)
        function_codes["CUDA_WRAPPER_FUNCTION_NAME"] = [get_function_name_from_cpp_or_cuda_code(wrapper) for wrapper in function_codes["CUDA_WRAPPER"]]
        # generate unit tests with source
        to_gen_directions = []
        to_gen_function_codes = []
        for i, pair in enumerate(translation_pairs):
            if pair.direction.source == "CPP":
                to_gen_directions.append(TransDirection(source="CPP"))
                to_gen_function_codes.append(pair.CPP)
            elif pair.direction.source == "CUDA":
                to_gen_directions.append(TransDirection(source="CUDA"))
                to_gen_function_codes.append(pair.CUDA)
            else:
                raise ValueError("Invalid source code")
        # TODO: fix duplicate generation of cuda wrapper 
        logger.info("Generating unit tests...")
        test_cases_list = self.generate_unit_tests(to_gen_function_codes, to_gen_directions, cached_cuda_wrapper=function_codes["CUDA_WRAPPER"])
        res = []
        for i, test_cases in enumerate(test_cases_list):
            if isinstance(test_cases, CppUnitTestModel):
                consistent_cpp_inputs = test_cases.test_cases
                consistent_cuda_inputs = self._replaces_wrapper_function_name(test_cases.test_cases, function_codes["CUDA_WRAPPER_FUNCTION_NAME"][i])
            elif isinstance(test_cases, CudaUnitTestModel):
                consistent_cuda_inputs = test_cases.test_cases
                consistent_cpp_inputs = self._replaces_wrapper_function_name(test_cases.test_cases, function_codes["CPP_FUNCTION_NAME"][i])
            else:
                raise ValueError("Invalid source code")
            case = UnitTestEvalCase(
                source = function_codes["DIRECTION"][i].source,
                cpp_code = function_codes["CPP"][i],
                cuda_code = function_codes["CUDA"][i],
                cuda_wrapper = function_codes["CUDA_WRAPPER"][i],
                consistent_cpp_inputs = consistent_cpp_inputs,
                consistent_cuda_inputs = consistent_cuda_inputs
            )
            res.append(case)
        return res
    
    
    def generate_test_files(
        self, function_codes: List[str], directions: List[TransDirection], output_dir: str = "unit_tests"
    ) -> List[str]:
        unit_tests = self.generate_unit_tests(function_codes, directions)
        test_cases_json_file_paths = []
        for i, unit_test in enumerate(unit_tests):
            save_dir = Path(output_dir) / f"{i}"
            test_cases_json_file_path = unit_test.generate_test_files(save_dir)
            test_cases_json_file_paths.append(test_cases_json_file_path)
        return test_cases_json_file_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--mode", type=str)

    args = parser.parse_args()
    # CUDA:
    tok_file_path = ROOT / "BabelTower" / "dataset" / "cuda.para.test.tok"
    lines = load_tok_file(tok_file_path)
    generator = UnitTestGenerator(model=args.model_path, mode=args.mode)  # 可选参数，默认是"gpt-4o"
    generator.generate_test_files(
        function_codes=lines, output_dir=args.output_dir, directions=[TransDirection(source="CUDA")]*len(lines)
    )

