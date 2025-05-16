from unit_test.compiler import CppCompiler, CudaCompiler
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from trans.utils.io import load_jsonlines, dump_json, load_tok_file, dump_jsonlines
from unit_test.templates import CPP_UNITTEST_TEMPLATES, CUDA_UNITTEST_TEMPLATES

def filter_cpp():
    cpp_mono = load_tok_file('', 'cpp', 'train')
    def compile_code(cpp):
        cpp_code = CPP_UNITTEST_TEMPLATES.replace('// TO_FILL_FUNC', cpp)
        compiler = CppCompiler()
        try:
            status, output, _ = compiler.run_code(cpp_code)
            if status == 'success':
                return cpp
        except Exception as e:
            print(f"Error: {str(e)}")
        return None

    results_code = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(compile_code, cpp) for cpp in cpp_mono]
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                results_code.append(future.result())
                print(len(results_code))
    results_json = [{'CPP' : code} for code in results_code]
    dump_jsonlines(results_json, '')

def filter_cuda():
    cuda_mono = load_tok_file('', 'cuda', 'train')
    def compile_code(cuda):
        cuda_code = CUDA_UNITTEST_TEMPLATES.replace('// KERNEL_FUNC', cuda)
        compiler = CudaCompiler()
        try:
            status, output, _ = compiler.run_code(cuda_code)
            if status == 'success':
                return cuda
        except Exception as e:
            print(f"Error: {str(e)}")
        return None
    
    results_code = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(compile_code, cuda) for cuda in cuda_mono]
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                results_code.append(future.result())
                print(len(results_code))
    results_json = [{'CUDA' : code} for code in results_code]
    dump_jsonlines(results_json, '')

if __name__ == '__main__':
    # filter_cpp()
    filter_cuda()