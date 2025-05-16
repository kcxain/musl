from unit_test.schemas import UnitTestEvalResultModel
from typing import Dict, Set, Tuple

def evaluate_test_cases(cpp_data: Dict, cuda_data: Dict) -> Tuple[Set, Set]:
    """
    Evaluate test cases and return sets of consistent and inconsistent cases
    """
    cpp_passed_cases = {k: v for k, v in cpp_data['status'].items() if v == 'success'}
    cuda_passed_cases = {k: v for k, v in cuda_data['status'].items() if v == 'success'}

    common_passed_cases = set(cpp_passed_cases.keys()) & set(cuda_passed_cases.keys())

    consistent_cases = set()
    inconsistent_cases = set()

    for case in common_passed_cases:
        if cpp_data['outputs'][case] == cuda_data['outputs'][case]:
            consistent_cases.add(case)
        else:
            inconsistent_cases.add(case)

    return consistent_cases, inconsistent_cases

def calculate_metrics(total_dirs: int, total_files: int, total_common_cases: int,
                     total_inconsistent_cases: int) -> UnitTestEvalResultModel:
    """
    Calculate evaluation metrics and return UnitTestEvalResultModel
    """
    compiling_pass_rate = total_files / total_dirs if total_dirs > 0 else 0
    running_pass_rate = total_common_cases / total_files if total_files > 0 else 0
    matching_rate = (total_common_cases - total_inconsistent_cases) / total_common_cases if total_common_cases > 0 else 0
    # Note: average_speedup_ratio would need actual timing data to calculate
    average_speedup_ratio = 1.0  # placeholder value

    return UnitTestEvalResultModel(
        compiling_pass_rate=compiling_pass_rate,
        running_pass_rate=running_pass_rate,
        matching_rate=matching_rate,
        average_speedup_ratio=average_speedup_ratio
    )