import importlib
import yaml
from pathlib import Path
from typing import Any

#This function was ai generated per description
def instantiate_component(config_dict: dict[str, Any]) -> Any:
    if "name" not in config_dict:
        raise ValueError("Configuration dictionary must contain 'name' for filters")

    class_path = config_dict.get("class_path")
    args = config_dict.get("args", {}) or {}

    if not class_path:
        raise ValueError("Configuration dictionary must contain a 'class_path'")
    
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ref = getattr(module, class_name)
    
    instance = class_ref(**args)
    return instance

def load_safety_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r") as file:
        raw_config = yaml.safe_load(file)
    
    raw_input_filters = raw_config.get("input_filters") or []
    raw_internal_filters = raw_config.get("internal_filters") or []
    raw_output_filters = raw_config.get("output_filters") or []
    
    return {
        "input_filters": [(instantiate_component(f), f["name"]) for f in raw_input_filters],
        "internal_filters": [(instantiate_component(f), f["name"]) for f in raw_internal_filters],
        "output_filters": [(instantiate_component(f), f["name"]) for f in raw_output_filters],
        "safety_config": raw_config.get("safety_config", {})
    }