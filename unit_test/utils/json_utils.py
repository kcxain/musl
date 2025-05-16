from pathlib import Path
import orjson as json
from typing import Any, Dict, List, Union
from pydantic import BaseModel


def load_json_file(file_path: Path) -> Any:
    """
    Load JSON data from a file.

    Args:
        file_path (Path): Path to the JSON file.

    Returns:
        Any: The Python representation of the JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.loads(file.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file {file_path}: {e}")


def save_json_file(file_path: Path, data: Any, indent: int = 2) -> None:
    """
    Save Python data to a JSON file.

    Args:
        file_path (Path): Path to save the JSON file.
        data (Any): The data to be serialized to JSON.
        indent (int, optional): Number of spaces to use for indentation. Defaults to 2.

    Raises:
        TypeError: If the data cannot be serialized to JSON.
        OSError: If there is an error writing the file.
    """
    try:
        with file_path.open("wb") as file:
            # `orjson.dumps` does not have a native indent option; you can mimic it:
            if indent > 0:
                formatted_json = json.dumps(data, option=json.OPT_INDENT_2)
            else:
                formatted_json = json.dumps(data)
            file.write(formatted_json)
    except TypeError as e:
        raise TypeError(f"Data cannot be serialized to JSON: {e}")
    except OSError as e:
        raise OSError(f"Failed to write to file {file_path}: {e}")


def save_jsonl_file(file_path: Path, data: Union[List[Dict], Dict, BaseModel], append: bool = False) -> None:
    """
    Save data to a JSONL file (JSON Lines format).

    Args:
        file_path (Path): Path to save the JSONL file.
        data: Data to save. Can be a single dict, list of dicts, or Pydantic BaseModel.
        append (bool): If True, append to existing file. If False, overwrite. Defaults to False.

    Raises:
        TypeError: If data format is invalid or cannot be serialized.
        OSError: If there is an error writing the file.
    """
    try:
        mode = "ab" if append else "wb"
        with file_path.open(mode) as file:
            if isinstance(data, BaseModel):
                file.write(json.dumps(data.model_dump()))
                file.write(b"\n")
            elif isinstance(data, dict):
                file.write(json.dumps(data))
                file.write(b"\n")
            elif isinstance(data, list):
                for item in data:
                    if not isinstance(item, (dict, BaseModel)):
                        raise TypeError("List items must be dictionaries or BaseModel instances")
                    if isinstance(item, BaseModel):
                        file.write(json.dumps(item.model_dump()))
                        file.write(b"\n")
                    else:
                        file.write(json.dumps(item))
                        file.write(b"\n")
            else:
                raise TypeError("Data must be a dictionary, list of dictionaries, or BaseModel instance")
    except TypeError as e:
        raise TypeError(f"Data cannot be serialized to JSONL: {e}")
    except OSError as e:
        raise OSError(f"Failed to write to file {file_path}: {e}")

def load_jsonl_file(file_path: Path, pydantic_model: Union[BaseModel, None]=None) -> Union[List[Dict], List[BaseModel]]:
    """
    Load data from a JSONL file (JSON Lines format).

    Args:
        file_path (Path): Path to the JSONL file.

    Returns:
        List[Dict]: List of dictionaries, each representing a JSON object from each line.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If any line contains invalid JSON.
        OSError: If there is an error reading the file.
    """
    result = []
    try:
        with file_path.open('rb') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data = json.loads(line)
                        if pydantic_model:
                            result.append(pydantic_model.model_validate(data))
                        else:
                            result.append(data)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in line: {line.decode('utf-8', errors='replace')}\nError: {e}")

        return result
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except OSError as e:
        raise OSError(f"Failed to read file {file_path}: {e}")