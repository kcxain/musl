from pydantic import BaseModel
import re
from loguru import logger

class GcovParser(BaseModel):
    function_name: str = None
    lines: int = 0
    branches: int = 0
    loops: int = 0
    lines_executed: float = 0.0
    branches_executed: float = 0.0
    taken: float = 0.0

def parse_gcov(gcov_data: str, func_name, gcov_path) -> GcovParser:
    # Regex patterns
    func_pattern = re.compile(rf"Function\s+'{func_name}\(.*\)'[\s\S]*?Lines\s+executed:(\d+\.\d+)%\s+of\s+(\d+)")

    branches_taken_pattern = re.compile(r"File\s+'\/tmp\/program.cpp'[\s\S]*?Branches\s+executed:(\d+\.\d+)%\s+of\s+(\d+)[\s\S]*?Taken\s+at\s+least\s+once:(\d+\.\d+)%\s+of\s+(\d+)")

    # Extract information
    func_match = func_pattern.search(gcov_data)
    branches_taken_match = branches_taken_pattern.search(gcov_data)

    if func_match:
        lines_executed = func_match.group(1)
        lines = func_match.group(2)

    if branches_taken_match:
        branches_executed = branches_taken_match.group(1)
        branches = branches_taken_match.group(2)
        taken_percentage = branches_taken_match.group(3)
        taken_total = branches_taken_match.group(4)
    try:
        ret = GcovParser(
            function_name=func_name,
            lines=int(lines),
            branches=int(branches),
            lines_executed=float(lines_executed),
            branches_executed=float(branches_executed),
            taken=float(taken_percentage)
        )
    except:
        logger.error(f"Error when parsing gcov for {func_name}")
        logger.info(gcov_path)
        return None

    return ret

