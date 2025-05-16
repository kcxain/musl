from pathlib import Path
import os


ROOT = Path(__file__).parent.parent
RESOURCES_DIR = ROOT / "resources"
OUTPUT_DIR = ROOT / "output"
DATA_DIR = ROOT / "data"
RESOURCES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
TRANSLATION_VALIDATION_OUTPUT_DIR = OUTPUT_DIR / "translation_validation"
TRANSLATION_VALIDATION_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

EVAL_TRANSLATED_OUTPUT_DIR = OUTPUT_DIR/ "eval_translated"
EVAL_TRANSLATED_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

EVAL_OUTPUTS_DIR = OUTPUT_DIR / "eval_outputs"


