import argparse
import os
import json
import subprocess
import tempfile
import concurrent.futures
from tqdm import tqdm

def compile_cuda(code):
    with tempfile.TemporaryDirectory() as temp_dir:
        cuda_file = os.path.join(temp_dir, "program.cu")
        exe_file = os.path.join(temp_dir, "program")

        with open(cuda_file, "w", encoding='utf-8') as f:
            f.write(code)

        try:
            compile_result = subprocess.run(
                [
                    "nvcc",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    cuda_file,
                    "-o",
                    exe_file
                ],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            if compile_result.returncode != 0:
                return False
            return True
        except:
            return False
        
def compile_cpp(code):
    with tempfile.TemporaryDirectory() as temp_dir:
        cpp_file = os.path.join(temp_dir, "program.cpp")
        exe_file = os.path.join(temp_dir, "program")

        with open(cpp_file, "w", encoding='utf-8') as f:
            f.write(code)

        try:
            # 编译
            compile_result = subprocess.run(
                ["g++", "-std=c++17", cpp_file, "-o", exe_file],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            if compile_result.returncode != 0:
                return False
            return True
        except:
            return False
                

def walk(dir):
    for root, _, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            file_name = file.split('.')[0]
            file_ext = file.split('.')[1]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                valid_lines = list(tqdm(executor.map(check, lines), total=len(lines), desc=f"Processing {file}"))
            valid_lines = [line for line, is_valid in zip(lines, valid_lines) if is_valid]
            new_file_path = os.path.join(root, f'{file_name}_filtered.{file_ext}')
            with open(new_file_path, 'w') as f:
                f.writelines(valid_lines)

def check(line):
    line_data = json.loads(line)
    from unit_test.templates import CPP_UNITTEST_TEMPLATES, CUDA_UNITTEST_TEMPLATES
    if line_data['source'] == 'CPP':
        code = line_data['CUDA']
        check_code = CUDA_UNITTEST_TEMPLATES.replace("// KERNEL_FUNC", code)
        return compile_cuda(check_code)
    elif line_data['source'] == 'CUDA':
        code = line_data['CPP']
        check_code = CPP_UNITTEST_TEMPLATES.replace("// TO_FILL_FUNC", code)
        return compile_cpp(check_code)
        

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dir', type=str, required=True)
    args = argparser.parse_args()
    walk(args.dir)