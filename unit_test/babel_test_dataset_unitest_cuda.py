from unit_test.configs import OUTPUT_DIR, ROOT
from unit_test.convertor import UnitTestConvertor
from unit_test.utils.json_utils import load_json_file
from unit_test.utils.common_utils import load_tok_file_as_map, get_unit_test_output_statistics
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Tuple
import threading
import argparse
from loguru import logger

# Create thread lock for counter synchronization
counter_lock = threading.Lock()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def convert_single_test(
    args: Tuple[UnitTestConvertor, int, str, list, Path]
) -> Tuple[int, bool, dict]:
    """
    Convert single test case with retry mechanism
    Returns: (case_id, is_success, failure_detail)
    """
    convertor, idx, cuda_code, unit_tests, output_dir = args
    try:
        save_dir = output_dir / f"{idx}"
        convertor.generate_test_files(
            target_kernel_code=cuda_code,
            test_cases=unit_tests,
            output_dir=save_dir
        )
        return idx, True, None
    except Exception as e:
        failure_detail = {
            "case_id": idx,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "cuda_code": cuda_code,
            "unit_tests": unit_tests
        }
        return idx, False, failure_detail

def update_progress(future, log_data, pbar):
    """Callback function to update progress and logs"""
    try:
        i, is_success, failure_detail = future.result()
        with counter_lock:
            if is_success:
                log_data["success_cases"] += 1
            else:
                log_data["failed_cases"] += 1
                log_data["failed_details"].append(failure_detail)
        pbar.update(1)
    except Exception as e:
        logger.error(f"Callback error: {e}")

def run(args, generated_tests_dir, cuda_code_map, log_data, max_workers):
    convertor = UnitTestConvertor()
    test_items = [item for item in generated_tests_dir.iterdir() if item.is_dir()]


    if "gpt" in args.model_path.lower():
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(test_items), desc="Converting CUDA tests") as pbar:
                futures = []
                for item in test_items:
                    idx = int(item.stem)
                    unit_tests = list(load_json_file(item / "test_cases.json")['cases'].values())
                    cuda_code = cuda_code_map[idx]
                    future = executor.submit(
                        convert_single_test,
                        (convertor, idx, cuda_code, unit_tests, args.output_dir)
                    )
                    future.add_done_callback(
                        lambda f, ld=log_data, p=pbar: update_progress(f, ld, p)
                    )
                    futures.append(future)

                concurrent.futures.wait(futures)
    else:
        for item in tqdm(test_items, desc="Converting CUDA tests"):
            idx = int(item.stem)
            unit_tests = list(load_json_file(item / "test_cases.json")['cases'].values())
            cuda_code = cuda_code_map[idx]

            idx, is_success, failure_detail = convert_single_test(
                (convertor, idx, cuda_code, unit_tests, args.output_dir)
            )
            if is_success:
                log_data["success_cases"] += 1
            else:
                log_data["failed_cases"] += 1
                log_data["failed_details"].append(failure_detail)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR / "babel_test_dataset_unitest_cuda")
    parser.add_argument("--model_path", type=str, default="gpt-4-turbo")

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True)

    generated_unit_tests_dir = OUTPUT_DIR / "babel_test_dataset_unitest"
    tok_file_path = ROOT / "BabelTower" / "dataset" / "cuda.para.test.tok"
    cuda_code_map = load_tok_file_as_map(tok_file_path)

    # Create log data dictionary
    log_data = {
        "total_cases": 0,
        "success_cases": 0,
        "failed_cases": 0,
        "failed_details": []
    }

    test_items = [item for item in generated_unit_tests_dir.iterdir() if item.is_dir()]
    log_data["total_cases"] = len(test_items)

    # Set maximum number of worker threads
    max_workers = min(8, len(test_items))

    run(args, generated_unit_tests_dir, cuda_code_map, log_data, max_workers)

    # Add timestamp
    log_data["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save log file
    log_file = args.output_dir / f"conversion_log_{log_data['timestamp']}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    # Print summary
    print("\nConversion Summary:")
    print(f"Total cases: {log_data['total_cases']}")
    print(f"Successful cases: {log_data['success_cases']}")
    print(f"Failed cases: {log_data['failed_cases']}")
    print(f"Log file saved to: {log_file}")

    get_unit_test_output_statistics(args.output_dir)

