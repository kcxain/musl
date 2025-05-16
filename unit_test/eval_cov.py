from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Tuple
import threading
import argparse
from unit_test.utils.json_utils import load_jsonl_file, save_jsonl_file
from unit_test.utils.parse_gcov import parse_gcov
from pydantic import BaseModel
from pathlib import Path
from unit_test.schemas import CppTestCovModel
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def statistic_test(data_pth: str = 'output/test_cov'):
    func_data = []
    gcov_paths = []
    code_paths = []
    # get all gcov files
    for root, dirs, files in os.walk(data_pth):
        for file in files:
            if file.endswith(".gcov"):
                gcov_path = os.path.join(root, file)
                gcov_paths.append(gcov_path)
            if file.endswith(".cpp"):
                code_path = os.path.join(root, file)
                code_paths.append(code_path)

    for gcov_path in tqdm(gcov_paths):
        with open(gcov_path, 'r') as f:
            gcov_data = f.read()
            func_name = gcov_path.split('.')[0].split('/')[-1]
            d = parse_gcov(gcov_data, func_name, gcov_path)
            if d:
                with open(code_paths[gcov_paths.index(gcov_path)], 'r') as f:
                    code = f.read()
                    loops = code.count('for') + code.count('while')
                    d.loops = loops
                func_data.append(d)

    return func_data


def babeltower_cov(data_pth: str, output_pth: str = 'test_cov') -> None:
    # Load the data
    raw_data = load_jsonl_file(Path(data_pth))
    cases = []
    for i, case in tqdm(enumerate(raw_data)):
        case_model = CppTestCovModel(
            function_code=case['cpp_code'],
            test_cases=case['consistent_cpp_inputs']
        )
        case_model.generate_test_files(f'{output_pth}/{i}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate unit tests for code')
    parser.add_argument('--data_pth', type=str, default='resources/unit_total_eval_cases.jsonl')
    parser.add_argument('--output_pth', type=str, default='output/test_cov')
    args = parser.parse_args()
    # babeltower_cov(args.data_pth)
    data = statistic_test()
    df = pd.DataFrame([d.dict() for d in data]) 
    df.to_excel("test_233.xlsx", index=False)
    save_jsonl_file(Path('test_cov.jsonl'), data)

    
