import os
import re
import gc
from openai import OpenAI
from openai.types.chat import ChatCompletion
from transformers import AutoTokenizer
from typing import List, Optional, Tuple, Literal, Union, Callable

from unit_test.configs import DOTENV_PATH
from unit_test.prompts import CUDA2CPP_TRANSLATION_PROMPT
from unit_test.utils.common_utils import remove_code_block_lines
from tenacity import retry, stop_after_attempt, wait_exponential
from enum import Enum
from trans.dataset import TransDirection
from loguru import logger
from dotenv import load_dotenv

load_dotenv(dotenv_path=DOTENV_PATH)

"""
    `*_trained` means the model has been fine-tuned on the dataset, so no need to parse output.
"""


class PromptType(str, Enum):
    UNITEST_TRAINED = 'test_trained'
    UNITEST = 'test'
    TRANS_TRAINED = 'trans_trained'
    TRANS = 'trans'
    CUDA_WRAPPER = 'cuda_wrapper'
    CUDA_WRAPPER_TRAINED = 'cuda_wrapper_trained'


class InferModel:
    def __init__(self, model_name: str, mode):
        self.model_name = model_name
        self.mode = mode
        self.trained = True if "trained" in mode else False
        # `None` when self.trained is True
        self.parse_output: Union[None, Callable] = None

    def collect_one(self, system: Optional[str], input: str, sample_num: int = 1) -> str:
        pass

    def collect_batch(self, systems: List[Optional[str]], inputs: List[str], sample_num: int = 1) -> List[str]:
        pass

    def generate_prompt(self, input: str, direction: Optional[TransDirection]) -> Tuple[str, str]:
        """
            TEST:
                - direction.source == 'CPP' means use 'CPP' code to generate tests
                - direction.source == 'CUDA' means use 'CUDA' code to generate tests
            TRANS:
                - direction.source == 'CPP' means trans 'CPP' to 'CUDA'
                - direction.source == 'CUDA' means trans 'CUDA' to 'CPP'
        """
        if self.mode == PromptType.UNITEST_TRAINED:
            from unit_test.prompts import CODE_BASED_CPP_UNIT_TEST_INPUT_GENERATION_PROMPT_TRAINED, CODE_BASED_CUDA_UNIT_TEST_INPUT_GENERATION_PROMPT_TRAINED
            if direction.source == 'CPP':
                system = CODE_BASED_CPP_UNIT_TEST_INPUT_GENERATION_PROMPT_TRAINED
            else:
                system = CODE_BASED_CUDA_UNIT_TEST_INPUT_GENERATION_PROMPT_TRAINED

            def _parse_output(output: str) -> List[str]:
                # Split by test case headers
                test_cases = re.split(r'//\s*Input case \d+:', output)
                # Remove empty strings and strip whitespace
                test_cases = [case.strip()
                              for case in test_cases if case.strip()]
                return test_cases
            self.parse_output = _parse_output
            return system, input

        elif self.mode == PromptType.TRANS_TRAINED:
            from trans.utils.prompts import CUDA_CPP_TRANSLATE_TRAIN_SYSTEM
            system = CUDA_CPP_TRANSLATE_TRAIN_SYSTEM.format(obj=direction)

            def _parse_output(output: str) -> List[str]:
                # TODO: parse output
                return output
            self.parse_output = _parse_output
            return system, input

        elif self.mode == PromptType.TRANS:
            if direction.source == 'CPP':
                from unit_test.prompts import CPP2CUDA_TRANSLATION_PROMPT
                prompt = CPP2CUDA_TRANSLATION_PROMPT.format(cpp_code=input)
            else:
                prompt = CUDA2CPP_TRANSLATION_PROMPT.format(cuda_code=input)

            def _parse_output(output: str) -> str:
                # 首先尝试匹配 [CODE] 和 [/CODE] 之间的内容
                pattern1 = r'\[CODE\](.*?)\[\/CODE\]'
                match1 = re.search(pattern1, output, re.DOTALL)

                if match1:
                    return remove_code_block_lines(match1.group(1).strip())

                # 如果第一个模式匹配失败，尝试匹配 ```cuda 和 ``` 之间的内容
                pattern2 = r'```cuda(.*?)```'
                match2 = re.search(pattern2, output, re.DOTALL)

                if match2:
                    return remove_code_block_lines(match2.group(1).strip())

                # 如果两种模式都匹配失败，抛出异常, 不是抛出异常，证明输出失败了
                return " "
                # raise ValueError("No code block found in the output")

            self.parse_output = _parse_output
            # no system
            return None, prompt

        elif self.mode == PromptType.UNITEST:
            from unit_test.prompts import CODE_BASED_CPP_UNIT_TEST_INPUT_GENERATION_PROMPT, CODE_BASED_CUDA_UNIT_TEST_INPUT_GENERATION_PROMPT
            if direction.source == 'CPP':
                prompt = CODE_BASED_CPP_UNIT_TEST_INPUT_GENERATION_PROMPT.format(
                    code=input)
            else:
                # when CUDA, input = "cuda_code\ncuda_wrapper
                prompt = CODE_BASED_CUDA_UNIT_TEST_INPUT_GENERATION_PROMPT.format(
                    code=input)

            def _parse_output(output: str) -> List[str]:
                # Match content between [TESTS] and [/TESTS]
                pattern = r"\[INPUTS\](.*?)\[/INPUTS\]"
                match = re.search(pattern, output, re.DOTALL)

                if not match:
                    return []

                test_content = match.group(1).strip()
                # Split by test case headers
                test_cases = re.split(r'//\s*Input case \d+:', test_content)
                # Remove empty strings and strip whitespace
                test_cases = [case.strip()
                              for case in test_cases if case.strip()]
                return test_cases

            self.parse_output = _parse_output
            # no system
            return None, prompt

        elif self.mode == PromptType.CUDA_WRAPPER:
            from unit_test.prompts import CUDA2CPP_WRAPPER_PROMPT
            prompt = CUDA2CPP_WRAPPER_PROMPT.format(cuda_code=input)

            def _parse_output(output: str) -> str:
                # 匹配 [CODE] 和 [/CODE] 之间的内容
                pattern = r'\[CODE\](.*?)\[\/CODE\]'

                # 使用re.search查找匹配
                # re.DOTALL让.能匹配换行符
                match = re.search(pattern, output, re.DOTALL)

                if match:
                    # group(1)获取第一个捕获组的内容
                    return remove_code_block_lines(match.group(1).strip())
                # logger.error(f"Failed to parse output\n{output}")
                return " "

            self.parse_output = _parse_output
            # no system
            return None, prompt
        elif self.mode == PromptType.CUDA_WRAPPER_TRAINED:
            from unit_test.prompts import CUDA2CPP_WRAPPER_PROMPT_TRAINED
            system = CUDA2CPP_WRAPPER_PROMPT_TRAINED

            def _parse_output(output: str) -> str:
                return output

            self.parse_output = _parse_output
            # no system
            return system, input
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


class OpenAIModel(InferModel):
    def __init__(self, model_name: str, mode):
        super().__init__(model_name=model_name, mode=mode)
        self.model = OpenAI().chat.completions
        self.pipe = self.model.create

    def post_process(self, output: ChatCompletion, n_sample=1) -> Union[str, List[str], List[List[str]]]:
        if (len(output.choices) == 1):
            # collect one
            return [self.parse_output(output.choices[0].message.content)]
        else:
            return [self.parse_output(out.message.content) for out in output.choices]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def collect_one(self, system: Optional[str], input: str, n_samples: int = 1) -> str:
        if system:
            message = [
                {"role": "system", "content": system},
                {"role": "user", "content": input},
            ]
        else:
            message = [
                {"role": "user", "content": input},
            ]
        params = {
            "temperature": 0.7,
            "top_p": 1.0,
        }
        if "o1" in self.model_name:
            params.pop("temperature")
        # logger.debug(f"messages: {message}")
        # stream = "qwen3-235b-a22b" == self.model_name
        res = self.pipe(
            model=self.model_name,
            n=n_samples,
            messages=message,
            seed=12345,
            # stream=stream,
            **params
        )  # .choices[0].message.content
        # logger.debug(f"Model: {self.model_name}, Input: {input}, Output: {res}")
        ret = self.post_process(res, n_samples)
        # logger.debug(f"Model: \n {self.model_name} \n Input: \n {input} \n params: \n {params} \n Output: \n {ret}")
        return ret
        # return self.parse_output(res)

    def collect_batch(self, systems: List[Optional[str]], inputs: List[str], n_samples: int = 1) -> List[str]:
        from tqdm.contrib.concurrent import thread_map
        results = thread_map(
            lambda x: self.collect_one(x[0], x[1], n_samples),
            list(zip(systems, inputs)),
            desc="Collecting"
        )
        ret = []
        for res in results:
            if isinstance(res, str):
                ret.append(res)
            else:
                ret.extend(res)
        return ret


class QWenModel(OpenAIModel):
    def __init__(self, model_name: str, mode):
        super().__init__(model_name=model_name, mode=mode)
        self.model = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).chat.completions
        self.pipe = self.model.create

    # qwen doesn't support batch n inference, we need to adjust it here.

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def collect_one(self, system: Optional[str], input: str, n_samples: int = 1) -> str:
        if system:
            message = [
                {"role": "system", "content": system},
                {"role": "user", "content": input},
            ]
        else:
            message = [
                {"role": "user", "content": input},
            ]
        params = {
            "temperature": 0.7,
            "top_p": 1.0,
        }
        if "o1" in self.model_name:
            params.pop("temperature")
        # logger.debug(f"messages: {message}")

        stream = "qwen3-235b-a22b" == self.model_name
        res = self.pipe(
            model=self.model_name,
            n=n_samples,
            messages=message,
            seed=12345,
            stream=stream,
            **params
        )  # .choices[0].message.content
        # logger.debug(f"Model: {self.model_name}, Input: {input}, Output: {res}")
        ret = self.post_process(res, n_samples)
        logger.debug(
            f"Model: \n {self.model_name} \n Input: \n {input} \n params: \n {params} \n Output: \n {ret}")
        return ret
        # return self.parse_output(res)


class DeepSeekModel(OpenAIModel):
    def __init__(self, model_name: str, mode):
        super().__init__(model_name=model_name, mode=mode)
        logger.debug(os.getenv("SILICON_FLOW"))
        self.model = OpenAI(
            api_key=os.getenv("SILICON_FLOW"),
            base_url="https://api.siliconflow.cn/v1",
        ).chat.completions
        self.pipe = self.model.create


class InternLmModel(OpenAIModel):
    def __init__(self, model_name: str, mode):
        super().__init__(model_name=model_name, mode=mode)
        self.model = OpenAI(
            api_key=os.getenv("INTERN_LM_API_KEY"),
            base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/",
        ).chat.completions
        self.pipe = self.model.create


class Qwen3LocalModel(InferModel):
    from unit_test.configs import TENSOR_PARRALLEL_SIZE
    try:
        from vllm import RequestOutput
    except ImportError:
        import warnings
        warnings.warn("vllm not installed, please install vllm first")
        from typing import Any
        RequestOutput = Any

    def __init__(self, model_name: str, mode: str, tensor_parallel_size: int = TENSOR_PARRALLEL_SIZE, enable_thinking: bool = False):
        super().__init__(model_name=model_name, mode=mode)
        from vllm import LLM
        self.model = LLM(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.pipe = self.model.generate
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def post_process(self, output: List[RequestOutput], n_sample=1) -> Union[str, List[str], List[List[str]]]:
        if len(output) == 1:
            # collect one
            return self.parse_output(output[0].outputs[0].text)
        else:
            # collect batch
            if n_sample == 1:
                return [self.parse_output(out.outputs[0].text) for out in output]
            else:
                # logger.info(f"{output}")
                return [self.parse_output(out.outputs[i].text) for out in output for i in range(n_sample)]

    def collect_one(self, system: str, input: str, n_sample: int = 1) -> Union[str, List[str]]:
        from vllm import SamplingParams
        # best practice, refer https://huggingface.co/Qwen/Qwen3-32B#best-practices
        sampling_params = SamplingParams(
            n=n_sample,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
            seed=42,
        )
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]
        text = self.tokenizer.apply_chat_template(
            conversation=message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Setting enable_thinking=False disables thinking mode
        )
        res = self.pipe(
            text,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        return self.post_process(res, n_sample)

    def collect_batch(self, systems: List[str], inputs: List[str], n_sample: int = 1) -> List[str]:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=n_sample,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
            seed=42,
        )
        messages = [
            [
                {"role": "system", "content": system},
                {"role": "user", "content": input},
            ]
            for system, input in zip(systems, inputs)
        ]
        texts = [self.tokenizer.apply_chat_template(
            conversation=message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Setting enable_thinking=False disables thinking mode
        ) for message in messages]

        for i, text in enumerate(texts):
            if len(self.tokenizer.encode(text)) > 20480:
                logger.warning(
                    f"Input text is too long: {text}, length: {len(self.tokenizer.encode(text))}")
                texts[i] = texts[i][:8192]

        res = self.pipe(
            prompts=texts,
            sampling_params=sampling_params,
            use_tqdm=True
        )
        return self.post_process(res, n_sample)

    def __del__(self):
        try:
            import torch
            import contextlib

            if torch.cuda.is_available():
                from vllm.distributed.parallel_state import (
                    destroy_model_parallel, destroy_distributed_environment
                )
                destroy_model_parallel()
                destroy_distributed_environment()
                del self.model.llm_engine.model_executor
                del self.model
                with contextlib.suppress(AssertionError):
                    torch.distributed.destroy_process_group()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            del self.model


class LocalModel(InferModel):
    from unit_test.configs import TENSOR_PARRALLEL_SIZE
    try:
        from vllm import RequestOutput
    except ImportError:
        import warnings
        warnings.warn("vllm not installed, please install vllm first")
        from typing import Any
        RequestOutput = Any

    def __init__(self, model_name: str, mode: str, tensor_parallel_size: int = TENSOR_PARRALLEL_SIZE):
        super().__init__(model_name=model_name, mode=mode)
        from vllm import LLM
        self.model = LLM(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.pipe = self.model.chat

    def post_process(self, output: List[RequestOutput], n_sample=1) -> Union[str, List[str], List[List[str]]]:
        if len(output) == 1:
            # collect one
            return self.parse_output(output[0].outputs[0].text)
        else:
            # collect batch
            if n_sample == 1:
                return [self.parse_output(out.outputs[0].text) for out in output]
            else:
                # logger.info(f"{output}")
                return [self.parse_output(out.outputs[i].text) for out in output for i in range(n_sample)]

    def collect_one(self, system: str, input: str, n_sample: int = 1) -> Union[str, List[str]]:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=n_sample,
            max_tokens=2048,
            seed=42,
        )
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]
        res = self.pipe(
            messages=message,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        return self.post_process(res, n_sample)

    def collect_batch(self, systems: List[str], inputs: List[str], n_sample: int = 1) -> List[str]:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=n_sample,
            max_tokens=2048,
            seed=42,
        )
        messages = [
            [
                {"role": "system", "content": system},
                {"role": "user", "content": input},
            ]
            for system, input in zip(systems, inputs)
        ]
        res = self.pipe(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=True
        )
        return self.post_process(res, n_sample)

    def __del__(self):
        try:
            import torch
            import contextlib

            if torch.cuda.is_available():
                from vllm.distributed.parallel_state import (
                    destroy_model_parallel, destroy_distributed_environment
                )
                destroy_model_parallel()
                destroy_distributed_environment()
                del self.model.llm_engine.model_executor
                del self.model
                with contextlib.suppress(AssertionError):
                    torch.distributed.destroy_process_group()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            del self.model


class ModelFactory:
    OPENAI_MODELS = ["openai", "gpt", "davinci", "o1"]
    # extra_body={"enable_thinking": True},
    QWEN_MODELS = ["qwen", "deepseek",
                   "qwen3-235b-a22b", "deepseek-r1", "deepseek-v3"]
    # DEEPSEEK_MODELS = ["deepseek", "Qwen/Qwen3-14B".lower(), "Qwen/Qwen3-8B".lower(), "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B".lower(),
    #                    "Pro/deepseek-ai/DeepSeek-V3".lower(), "Pro/deepseek-ai/DeepSeek-R1".lower(), "Qwen/Qwen3-235B-A22B".lower(),
    #                    "THUDM/GLM-Z1-32B-0414".lower()]
    INTERN_LM_MODELS = ["internlm"]

    @staticmethod
    def get_model(model_name: str, mode: PromptType, tensor_parallel_size: int = 4):
        # trained: 是否已经 SFT 的模型，所需要的 Prompt 会不同
        if os.path.exists(model_name) and "qwen3" in model_name.lower():
            return Qwen3LocalModel(model_name, mode, tensor_parallel_size)
        if os.path.exists(model_name):
            return LocalModel(model_name, mode, tensor_parallel_size)
        if any(model in model_name.lower() for model in ModelFactory.OPENAI_MODELS):
            return OpenAIModel(model_name, mode)
        # elif any(model in model_name.lower() for model in ModelFactory.DEEPSEEK_MODELS):
        #     return DeepSeekModel(model_name, mode)
        elif any(model in model_name.lower() for model in ModelFactory.QWEN_MODELS):
            return QWenModel(model_name, mode)
        elif any(model in model_name.lower() for model in ModelFactory.INTERN_LM_MODELS):
            return InternLmModel(model_name, mode)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
