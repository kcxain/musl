from unit_test.configs import ROOT, OUTPUT_DIR
from unit_test.generator import UnitTestGenerator
from unit_test.utils.common_utils import load_tok_file, count_nonempty_lines, get_unit_test_output_statistics
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Tuple
import threading
import argparse
from unit_test.utils.json_utils import load_json_file

from unit_test.schemas import CppUnitTestModel
from trans.dataset import TransDirection
from models import ModelFactory
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 创建线程锁用于同步计数器更新
counter_lock = threading.Lock()

def execute_single_test(
    args: Tuple[CppUnitTestModel, int, str, str]
) -> Tuple[int, bool, dict]:
    """
    执行单元测试任务
    返回: (case_id, is_success, failure_detail)
    """
    unit_test, i, function_code, output_dir = args
    failure_details = []
    try:
        save_dir = output_dir / f"{i}"
        json_path=unit_test.generate_test_files(output_dir=save_dir)
        with open(json_path, 'r') as f:
            data = json.load(f)
        if "status" in data and "outputs" in data:
            status_dict = data["status"]
            for key, value in status_dict.items():
                if value != "success":
                    failure_details.append({
                        "case_id": i,
                        "error_type": value,
                        "error_message": data["outputs"][key],
                        "function_code": function_code,
                    })
        if failure_details:
            return i, False, failure_details
        else:
            return i, True, None
    except Exception as e:
        failure_detail = [{
            "case_id": i,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "function_code": function_code,
        }]
        return i, False, failure_detail


def update_progress(future, log_data, pbar):
    """回调函数，用于更新进度和日志"""
    try:
        i, is_success, failure_detail = future.result()
        with counter_lock:
            if is_success:
                log_data["success_cases"] += 1
                log_data["success_ids"].append(i)
            else:
                log_data["failed_cases"] += 1
                log_data["failed_details"].extend(failure_detail)
        pbar.update(1)
    except Exception as e:
        print(f"Callback error: {e}")


def run(model, mode, lines, directions, log_data, max_workers, output_dir):
    generator = UnitTestGenerator(model=model, mode=mode)
    cached_cuda_wrapper = load_json_file(Path('/lustre/S/wangshuohit/Project/self-training/BabelTower/generated_data/test_cached_cuda_wrapper.json'))["wrapper"]
    unit_tests = generator.generate_unit_tests(lines,directions,cached_cuda_wrapper)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建进度条
        with tqdm(total=len(lines), desc="Generating unit tests") as pbar:
            # 提交所有任务
            futures = []
            for i, (unit_test, function_code) in enumerate(zip(unit_tests, lines)):
                future = executor.submit(
                    execute_single_test,
                    (unit_test, i, function_code, output_dir),
                )
                future.add_done_callback(
                    lambda f, ld=log_data, p=pbar: update_progress(f, ld, p)
                )
                futures.append(future)

            # 等待所有任务完成
            concurrent.futures.wait(futures)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default= OUTPUT_DIR / "babel_test_dataset_unitest")
    parser.add_argument("--model_path", type=str, default="gpt-4")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    # 创建日志记录字典
    log_data = {
        "total_cases": 0,
        "success_cases": 0,
        "success_ids": [],
        "failed_cases": 0,
        "failed_details": [],
    }

    if args.cuda:
        tok_file_path = ROOT / "BabelTower" / "dataset" / "cuda.para.test.tok"
    else:
        tok_file_path = ROOT / "BabelTower" / "dataset" / "cpp.para.test.tok"
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True)
    lines = load_tok_file(tok_file_path)
    no_empty_length = count_nonempty_lines(tok_file_path)
    assert len(lines) == no_empty_length

    log_data["total_cases"] = len(lines)

    # 设置线程池最大工作线程数
    max_workers = min(8, len(lines))  # 限制最大线程数为8

    if args.cuda:
        directions=[TransDirection(source="CUDA")]*len(lines)
    else:
        directions=[TransDirection(source="CPP")]*len(lines)

    run(args.model_path, args.mode, lines, directions, log_data, max_workers, args.output_dir)

    # 添加时间戳
    log_data["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 保存日志文件
    log_file = args.output_dir / f"generation_log_{log_data['timestamp']}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    # 打印总结
    print("\nGeneration Summary:")
    print(f"Successful cases: {log_data['success_cases']}/{log_data['total_cases']}")
    print(f"Failed cases: {log_data['failed_cases']}/{log_data['total_cases']}")
    print(f"Log file saved to: {log_file}")

    get_unit_test_output_statistics(args.output_dir)

