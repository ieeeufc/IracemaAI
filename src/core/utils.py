from typing import Any, Dict

import yaml


def read_yaml_file(file_name: str) -> Dict[str, Any]:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
            import logging
            logging.error(exc)

    Returns:
        Dict[str, Any]: The contents of the YAML file as a dictionary.
    """
    with open(file_name, 'r', encoding='utf-8') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return {}
