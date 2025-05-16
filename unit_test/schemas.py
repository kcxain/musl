import json

from pydantic import BaseModel
from enum import Enum
from typing import List, Union, Literal
from dataclasses import dataclass

import os
from tqdm import tqdm
from unit_test.compiler import CppCompiler, CudaCompiler
from unit_test.configs import ENABLE_COMPILING_WHEN_GENERATING_UNIT_TESTS
from unit_test.templates import CPP_UNITTEST_TEMPLATES, CUDA_UNITTEST_TEMPLATES, CPP_UNITTEST_TEMPLATES_FOR_COV
from unit_test.utils.common_utils import replace_wrapper_func_first_arg, replace_wrapper_invoke, get_function_name_from_cpp_or_cuda_code

from trans.dataset import TransDirection
from models import ModelFactory, InferModel, PromptType

from multiprocessing import Pool
import multiprocessing
from pathlib import Path
from functools import partial

class UnitTestModel(BaseModel):
    pass


# g++ -std=c++17 test.cpp -o test && ./test
class CppUnitTestModel(UnitTestModel):
    template: str = CPP_UNITTEST_TEMPLATES
    raw_test_cases: Union[List[str], None] = None

    function_code: str
    test_cases: List[str]

    def _process_test_case(self, test_case_tuple, output_dir, template_data):
        i, test_case = test_case_tuple

        # 生成cpp代码
        code = template_data["template"].replace("// TO_FILL_FUNC", template_data["function_code"])
        code = code.replace("// TEST_CASE", test_case)

        # 生成文件名
        cpp_file_name = f"test_case_{i + 1}.cpp"
        cpp_file_path = os.path.join(output_dir, cpp_file_name)

        # 写入cpp文件
        with open(cpp_file_path, "w", encoding='utf-8') as f:
            f.write(code)

        result = {
            "case": test_case,
            "output": None,
            "status": None
        }

        if ENABLE_COMPILING_WHEN_GENERATING_UNIT_TESTS:
            compiler = CppCompiler()
            status, output, _ = compiler.run_code(code)
            result["output"] = output
            if output=="":
                status = "invalid_test_case"
            result["status"] = status

        return i, result

    def generate_test_files(self, output_dir: str = "unit_tests") -> str:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 准备模板数据
        template_data = {
            "template": self.template,
            "function_code": self.function_code
        }

        # 创建进程池
        num_processes = multiprocessing.cpu_count()
        pool = Pool(processes=num_processes)

        # 使用偏函数固定某些参数
        process_func = partial(self._process_test_case,
                               output_dir=output_dir,
                               template_data=template_data)

        # 并行处理所有test cases
        results = list(tqdm(
            pool.imap(process_func, enumerate(self.test_cases)),
            total=len(self.test_cases),
            desc="Generating test files"
        ))

        pool.close()
        pool.join()

        # 整理结果
        json_data = {
            "function_name": self.function_code,
            "cases": {},
            "outputs": {},
            "status": {},
        }

        for i, result in results:
            case_id = str(i + 1)
            json_data["cases"][case_id] = result["case"]
            json_data["outputs"][case_id] = result["output"]
            json_data["status"][case_id] = result["status"]

        # 写入json文件
        json_file_path = os.path.join(output_dir, "cpp_test_cases.json")
        with open(json_file_path, "w", encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

        return json_file_path


class UnitTestEvalCase(BaseModel):
    cpp_code: str
    cuda_code: str
    consistent_cpp_inputs: List[str]
    consistent_cuda_inputs: List[str]
    cuda_wrapper: str
    source: str = "None"

    def format_cuda_code(self) -> List[str]:
        cuda_unittest_codes = []
        for test_case in self.consistent_cuda_inputs:
            code = CUDA_UNITTEST_TEMPLATES.replace("// KERNEL_FUNC", self.cuda_code)
            code = code.replace("// WRAPPER_FUNC", self.cuda_wrapper)
            code = code.replace("// TEST_CASE", test_case)
            cuda_unittest_codes.append(code)
        return cuda_unittest_codes

    def format_cpp_code(self) -> List[str]:
        cpp_unittest_codes = []
        for test_case in self.consistent_cpp_inputs:
            code = CPP_UNITTEST_TEMPLATES.replace("// TO_FILL_FUNC", self.cpp_code)
            code = code.replace("// TEST_CASE", test_case)
            cpp_unittest_codes.append(code)
        return cpp_unittest_codes


class CudaUnitTestModel(UnitTestModel):
    template: str = CUDA_UNITTEST_TEMPLATES
    kernel_code_function_name: str
    wrapper_function_function_name: str
    kernel_code: str
    wrapper_function: str
    raw_test_cases: Union[List[str], None] = None
    test_cases: List[str]

    def _process_test_case(self, test_case_tuple, output_dir, template_data):
        i, test_case = test_case_tuple

        # replace the function name of test_case to wrapper_function_function_name
        test_case = replace_wrapper_func_first_arg(
            test_case,
            template_data["wrapper_function_function_name"]
        )

        # 生成cuda代码
        code = template_data["template"].replace("// KERNEL_FUNC", template_data["kernel_code"])
        code = code.replace("// WRAPPER_FUNC", template_data["wrapper_function"])
        code = code.replace("// TEST_CASE", test_case)

        # DEBUG 生成文件名
        cuda_file_name = f"test_case_{i + 1}.cu"
        cuda_file_path = os.path.join(output_dir, cuda_file_name)

        # 写入cuda文件
        with open(cuda_file_path, "w", encoding='utf-8') as f:
            f.write(code)

        result = {
            "case": test_case,
            "output": None,
            "status": None
        }

        if ENABLE_COMPILING_WHEN_GENERATING_UNIT_TESTS:
            compiler = CudaCompiler()
            status, output, _ = compiler.run_code(code)
            # TODO to be improved
            if any(not template_data[key].strip() for key in ["wrapper_function", "kernel_code"]):  # 无效的代码
                status = "invalid_wrapper"
            elif not test_case.strip():  # 无效的测试用例
                status = "invalid_test_case"
            result["output"] = output
            result["status"] = status

        return i, result

    def generate_test_files(self, output_dir: str = "unit_tests") -> str:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # 准备模板数据
        template_data = {
            "template": self.template,
            "kernel_code": self.kernel_code,
            "wrapper_function": self.wrapper_function,
            "wrapper_function_function_name": self.wrapper_function_function_name
        }

        # 创建进程池
        num_processes = multiprocessing.cpu_count()
        pool = Pool(processes=num_processes)

        # 使用偏函数固定某些参数
        process_func = partial(self._process_test_case,
                               output_dir=output_dir,
                               template_data=template_data)

        # 并行处理所有test cases
        results = list(tqdm(
            pool.imap(process_func, enumerate(self.test_cases)),
            total=len(self.test_cases),
            desc="Generating test files"
        ))

        pool.close()
        pool.join()

        # 整理结果
        json_data = {
            "function_name": self.kernel_code,
            "wrapper_function": self.wrapper_function,
            "cases": {},
            "outputs": {},
            "status": {},
        }

        for i, result in results:
            case_id = str(i + 1)
            json_data["cases"][case_id] = result["case"]
            json_data["outputs"][case_id] = result["output"]
            json_data["status"][case_id] = result["status"]

        # 写入json文件
        json_file_path = output_dir / "cuda_test_cases.json"
        with open(json_file_path, "w", encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

        return str(json_file_path)

class CppTestCovModel(UnitTestModel):
    template: str = CPP_UNITTEST_TEMPLATES_FOR_COV
    function_code: str
    test_cases: List[str]

    def generate_test_files(self, output_dir: str = "unit_tests", save_file: bool = True) -> str:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 准备模板数据
        template_data = {
            "template": self.template,
            "function_code": self.function_code
        }
        # 生成cpp代码
        code = template_data["template"].replace("// TO_FILL_FUNC", template_data["function_code"])

        test_cases_all = [replace_wrapper_invoke(case) for case in self.test_cases]
        test_cases_all = "\n".join(test_cases_all)
        code = code.replace("// TEST_CASE", test_cases_all)
        func_name = get_function_name_from_cpp_or_cuda_code(template_data["function_code"])
        # 生成文件名
        cpp_file_name = f"{func_name}.cpp"
        cpp_file_path = os.path.join(output_dir, cpp_file_name)

        # 写入cpp文件
        if save_file:
            with open(cpp_file_path, "w", encoding='utf-8') as f:
                f.write(code)

        compiler = CppCompiler()
        status, output, gcov_out, _ = compiler.run_code_with_cov(code)
        if output=="":
            status = "invalid_test_case"
        if save_file:
            gcov_file_path = os.path.join(output_dir, f"{func_name}.gcov")
            with open(gcov_file_path, "w", encoding='utf-8') as f:
                f.write(gcov_out)


class TranslationPair(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True  # 允许任意类型
    }
    CUDA: str
    CPP: str
    direction: TransDirection


class TranslationInput(BaseModel):
    model_config = {"protected_namespaces": ()}  # Disable protected namespace checking
    model_path: str  # 模型路径
    input_file: str  # 输入文件
    mode: Literal[PromptType.UNITEST, PromptType.UNITEST_TRAINED]  # only unit test prompt is supported
    translation_pairs: Union[List[TranslationPair], None] = None  # 翻译对列表

    def get_model(self) -> InferModel:
        """
        Get the translation model from the model path.
        """
        model_factory = ModelFactory()
        return model_factory.get_model(self.model_path, mode=self.mode)

    def load_from_jsonl(self) -> None:
        """
        Load translation pairs from a JSONL file and populate the translation_pairs field.
        """
        translations = []

        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())

                        cuda_code = data.get("CUDA")
                        cpp_code = data.get("CPP")
                        source_type = data.get("source", "")

                        if not cuda_code or not cpp_code:
                            continue
                        
                        direction = TransDirection(source=source_type)

                        translation = TranslationPair(
                            CUDA=cuda_code,
                            CPP=cpp_code,
                            direction=direction
                        )
                        translations.append(translation)

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error processing line: {e}")
                        continue

            self.translation_pairs = translations

        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {self.input_file} not found")
        except Exception as e:
            raise Exception(f"Error loading translations: {e}")


# 输出数据模型
class ValidTranslation(BaseModel):
    CUDA: str
    CPP: str
    CUDA_WRAPPER: str
    source: str
    valid_unit_tests_inputs: List[str]  # 合法的输入列表


class UnitTestEvalResultModel(BaseModel):
    model_config = {
        "protected_namespaces": ()  # 允许任意类型
    }
    model_to_eval: str = ""  # 空字符串作为默认值
    total_functions: int = 0
    total_cases: int = 0
    compiling_pass_rate: float = 0.0
    running_pass_rate: float = 0.0
    matching_rate_cases: float = 0.0
    matching_rate_functions: float = 0.0
    average_speedup_ratio: float = 1.0

if __name__ == "__main__":
    # 使用示例
    cpp_unit_test = CppUnitTestModel(
        function_code="""void sum_backward ( float * db , float * dout , int r , int c ) { 
        for ( int j = 0 ; j < c ; j ++ ) { 
            for ( int i = 0 ; i < r ; i ++ ) { 
                db [ j ] += dout [ i * c + j ] ; 
            } 
        } 
    }""",
        test_cases=parsed_output,
    )

    cpp_code = cpp_unit_test.generate_code()

    # 将生成的代码写入文件
    with open("unit_test.cpp", "w", encoding='utf-8') as f:
        f.write(cpp_code)
