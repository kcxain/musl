import os
from pathlib import Path
from unit_test.utils.common_utils import (
    get_function_name_from_cpp_or_cuda_code,
    replace_kernel_function_in_wrapper,
    replace_wrapper_func_first_arg,
    replace_wrapper_invoke_back,
    remove_comments
)
from unit_test.utils.json_utils import load_json_file, load_jsonl_file, save_json_file, save_jsonl_file
from unit_test.prompts import CODE_BASED_CPP_UNIT_TEST_INPUT_GENERATION_PROMPT_TRAINED \
, CODE_BASED_CUDA_UNIT_TEST_INPUT_GENERATION_PROMPT_TRAINED, CUDA2CPP_WRAPPER_PROMPT_TRAINED

prompt_dict = {
    'CPP':CODE_BASED_CPP_UNIT_TEST_INPUT_GENERATION_PROMPT_TRAINED,
    'CUDA':CODE_BASED_CUDA_UNIT_TEST_INPUT_GENERATION_PROMPT_TRAINED
}
def build_dataset_ut(input_dir,output_path,shuffle=False):
    ''' 从筛选后的数据构建单测数据集 '''
    output = []
    for file in os.listdir(input_dir):
        num = 0
        inputs=load_jsonl_file(Path(input_dir) / file)
        for item in inputs:
            target = 'CPP' if item['source']=='CUDA' else 'CUDA'
            function_name = get_function_name_from_cpp_or_cuda_code(item[target if target=='CPP' else 'CUDA_WRAPPER'])
            temp_dict = {}
            unit_tests = ''
            if item['valid_unit_tests_inputs'] == [] or item['valid_unit_tests_inputs'][0] == ' ' or item['CUDA_WRAPPER'] == ' ':
                continue
            for i,case in enumerate(item['valid_unit_tests_inputs']):
                replaced = replace_wrapper_func_first_arg(case, function_name)
                replaced = replace_wrapper_invoke_back(replaced, function_name)
                unit_tests += f"\n//Input case {i+1}:\n"+replaced
            temp_dict["system"] = prompt_dict[target]
            temp_dict["instruction"] = item[target] if target=='CPP' else item[target]+"\n"+item['CUDA_WRAPPER']
            temp_dict["output"] = unit_tests[1:] # remove the prefix '\n'
            temp_dict["target"] = target # 希望本数据加强模型对target语言生成单测的能力
            output.append(temp_dict)
            num +=1
        print(file, num)
    if shuffle:
        import random
        random.shuffle(output)
    save_json_file(Path(output_path), output)

def build_dataset_ut_flitered(file_path_old,file_path_new,output_path,ratio=0.3,shuffle=False):
    data_old = load_json_file(Path(file_path_old))
    data_new = load_json_file(Path(file_path_new))
    old_instruction = [item['instruction'][:60] for item in data_old] # wrapper 可能不一样，只看前缀
    flitered_data = []
    for item in data_new:
        if item['instruction'][:60] in old_instruction:
            import random
            if random.random() < ratio:
                flitered_data.append(item)
        else:
            flitered_data.append(item)
    if shuffle:
        import random
        random.shuffle(flitered_data)
    print(file_path_old, len(data_old))
    print(file_path_new, len(data_new))
    print(output_path, len(flitered_data))
    save_json_file(Path(output_path), flitered_data)

def build_dataset_wrapper(input_path,output_path,shuffle=False):
    ''' 从筛选后的数据构建wrapper数据集 '''
    output = []
    num = 0
    inputs=load_jsonl_file(Path(input_path))
    for item in inputs:
        temp_dict = {}
        temp_dict["system"] = CUDA2CPP_WRAPPER_PROMPT_TRAINED
        temp_dict["instruction"] = item["cuda_code"]
        temp_dict["output"] = remove_comments(item["cuda_wrapper"])
        output.append(temp_dict)
        num +=1
    print(num)
    if shuffle:
        import random
        random.shuffle(output)
    save_json_file(Path(output_path), output)


def build_dataset_wrapper_gpt(input_path,output_path,shuffle=False):
    ''' 用GPT生成wrapper数据集 '''
    # from models import ModelFactory
    # from unit_test.configs import ROOT, CUDA_WRAPPER_MODEL_PATH, CUDA_WRAPPER_MODEL_PATH_TRAINED
    # cuda_wrapper_model = ModelFactory.get_model(model_name='gpt-4-turbo', mode='cuda_wrapper')
    # # cuda_wrapper_model = ModelFactory.get_model(CUDA_WRAPPER_MODEL_PATH, mode='cuda_wrapper')
    # inputs=load_jsonl_file(Path(input_path))
    # cuda_codes=[item["CUDA"] for item in inputs]
    # import random
    # random.seed(0)
    # random_index = random.sample(range(len(inputs)), 3000)
    # cuda_codes = [cuda_codes[i] for i in random_index]
    # wrapper_systems, wrapper_prompts = [], []
    # for cuda_code in cuda_codes:
    #     wrapper_system, wrapper_prompt = cuda_wrapper_model.generate_prompt(cuda_code, None)
    #     wrapper_systems.append(wrapper_system)
    #     wrapper_prompts.append(wrapper_prompt)
    # wrapper_cudas = cuda_wrapper_model.collect_batch(systems=wrapper_systems, inputs=wrapper_prompts)
    # oringin_data = []
    # for a,b in zip(cuda_codes,wrapper_cudas):
    #     oringin_data.append({"cuda_code":a,"cuda_wrapper":b})
    # save_json_file(Path(output_path+".ori.json"), oringin_data)

    ''' 对生成的wrapper编译 '''
    # NOTE: in some reason, compile the generated wrapper code is another task
    data = load_json_file(Path(output_path+".ori.json"))
    cuda_codes=[item["cuda_code"] for item in data]
    wrapper_cudas=[item["cuda_wrapper"] for item in data]
    # NOTE: to avoid generate wrong function name
    for i in range(len(wrapper_cudas)):
        wrapper_cudas[i] = replace_kernel_function_in_wrapper(wrapper_cudas[i], get_function_name_from_cpp_or_cuda_code(cuda_codes[i]))
    from unit_test.compiler import CudaCompiler
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from unit_test.templates import CUDA_UNITTEST_TEMPLATES
    def filter_cuda(cuda_list,wrapper_list):
        def compile_code(cuda,wrapper):
            if wrapper == ' ':
                print('wrapper is empty')
                return None
            cuda_code = CUDA_UNITTEST_TEMPLATES.replace('// KERNEL_FUNC', cuda)
            cuda_code = cuda_code.replace('// WRAPPER_FUNC', wrapper)
            compiler = CudaCompiler()
            try:
                status, output, _ = compiler.run_code(cuda_code)
                if status == 'success':
                    return [cuda, wrapper]
            except Exception as e:
                print(f"Error: {str(e)}")
            return None
        
        results_code = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(compile_code, cuda, wrapper) for cuda,wrapper in zip(cuda_list,wrapper_list)]
            for future in tqdm(as_completed(futures), total=len(futures)):
                if future.result():
                    temp_dict = {}
                    temp_dict["system"] = CUDA2CPP_WRAPPER_PROMPT_TRAINED
                    temp_dict["instruction"] = future.result()[0]
                    temp_dict["output"] = future.result()[1]
                    results_code.append(temp_dict)
        return results_code

    output = filter_cuda(cuda_codes,wrapper_cudas)
    if shuffle:
        import random
        random.shuffle(output)
    print(len(output))
    save_json_file(Path(output_path), output)

if __name__ == '__main__':