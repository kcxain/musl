import json
import random
import argparse
from pathlib import Path

from openai import OpenAI
from typing import List
from loguru import logger

from unit_test.configs import ROOT, OUTPUT_DIR
from unit_test.prompts import (
    CUDA2CPP_WRAPPER_PROMPT,
    CODE_BASED_CPP_UNIT_TEST_INPUT_GENERATION_PROMPT
)
import re

from unit_test.schemas import CppUnitTestModel, CudaUnitTestModel, CppConvertedCudaWrapperModel
from unit_test.utils.common_utils import (
    get_function_name_from_cpp_or_cuda_code,
    replace_assert_statements,
    wrapper_function_invoke_to_print_variables,
    load_tok_file,
    replace_wrapper_invoke_back,
    replace_cuda_free_statements, remove_code_block_lines, load_tok_file_as_map,
)


class UnitTestConvertor:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        if "gpt" in self.model:
            self.client = OpenAI()
            self.pipe = self.client.chat.completions.create
        else:
            from transformers import pipeline
            self.pipe = pipeline("text-generation", model=self.model, device_map="auto")

    def convert_unit_tests(
            self, target_kernel_code: str
    ) -> CppConvertedCudaWrapperModel:
        wrapper_conversion_prompt = WrapperConversionPrompt(
            target_kernel_code=target_kernel_code,
        )
        prompt = wrapper_conversion_prompt.generate()
        messages = [
            {"role": "system", "content": prompt},
        ]
        # logger.debug(f"prompt: \n{prompt}")

        if "gpt" in self.model:
            completion = self.pipe(model=self.model, messages=messages)
            wrapper_code_block = completion.choices[0].message.content
        else:
            completion = self.pipe(messages)
            # system, user, assistant
            wrapper_code_block = completion[0][0]["generated_text"][2]["content"]
        # logger.debug(wrapper_code_block)

        wrapper_function = wrapper_conversion_prompt.parse_output(wrapper_code_block)
        # 这里生成的就不是test cases了，  对应的CPP unitest 的cuda的wrapper 版本
        wrapper_model = CppConvertedCudaWrapperModel(
            kernel_code=target_kernel_code,
            wrapper_function=wrapper_function,
            kernel_code_function_name=get_function_name_from_cpp_or_cuda_code(target_kernel_code),
            wrapper_function_function_name=get_function_name_from_cpp_or_cuda_code(wrapper_function),
        )

        return wrapper_model

    def generate_test_files(
            self,
            target_kernel_code: str,
            test_cases: List[str],  # this is compatible with both the cuda and the cpp unit tests.
            output_dir: str = "unit_tests",
    ) -> str:
        cuda_unit_test = self.convert_unit_tests(
            target_kernel_code
        )
        cuda_unit_test_model = CudaUnitTestModel(
            kernel_code=target_kernel_code,
            wrapper_function=cuda_unit_test.wrapper_function,
            kernel_code_function_name=cuda_unit_test.kernel_code_function_name,
            wrapper_function_function_name=cuda_unit_test.wrapper_function_function_name,
            test_cases=test_cases,

        )

        # def generate_test_files(self, output_dir: str = "unit_tests") -> None:
        unit_test_result_json_path = cuda_unit_test_model.generate_test_files(output_dir=output_dir)
        return unit_test_result_json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert unit tests from CPP to CUDA')
    parser.add_argument('--idx', type=int, default=11,
                        help='Index of the sample to process (default: 10)')
    parser.add_argument('--output_dir', type=str, default='converted_unit_tests',
                        help='Output directory for generated test files')
    args = parser.parse_args()

    sampled_idx: int = args.idx
    tok_file_path = ROOT / "BabelTower" / "dataset" / "cuda.para.test.tok"
    source_test_cases_json = (
            OUTPUT_DIR / "babel_test_dataset_unitest" /
            str(sampled_idx) / "test_cases.json"
    )

    # Load and format test cases
    data = json.load(open(source_test_cases_json, "r"))
    # 有解析错误， 不能这样干。
    cases = list(data["cases"].values())
    # TODO: 这里的映射关系还有问题， 要改一下， 不是简单的用文件名就行了， 应该还是索引， 那么就要保证就算失败了 索引应该还是对的
    function_name = get_function_name_from_cpp_or_cuda_code(data["function_name"])
    # Load CUDA kernel code map
    target_kernel_code = load_tok_file(tok_file_path)[args.idx]

    # Initialize converter and generate test files
    convertor = UnitTestConvertor()
    output_dir = Path(args.output_dir) / str(sampled_idx)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate test files
        convertor.generate_test_files(
            target_kernel_code=target_kernel_code,
            test_cases=cases,
            output_dir=str(output_dir),
        )
    except Exception as e:
        logger.error(f"Error generating test files: {str(e)}")
        from traceback import print_exc

        print_exc()
