import importlib
import yaml
from pathlib import Path
from typing import Any

#This function was ai generated per description
def instantiate_component(config_dict: dict[str, Any]) -> Any:
    """Instantiates the class at runtime """
    if "name" not in config_dict:
        raise ValueError("Configuration dictionary must contain 'name' for detectors")

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
    """ Loads the Safety Harness configuration to apply around the LLM

    Args:
        config_path: Path to the config yaml

    Returns:
        dict containing the instantiated safety classifier and metadata.
    """
    with open(config_path, "r") as file:
        raw_config = yaml.safe_load(file)
    
    raw_input_detectors = raw_config.get("input_detectors") or []
    raw_internal_detectors = raw_config.get("internal_detectors") or []
    raw_output_detectors = raw_config.get("output_detectors") or []
    
    return {
        "input_detectors": [(instantiate_component(f), f["name"]) for f in raw_input_detectors],
        "internal_detectors": [(instantiate_component(f), f["name"]) for f in raw_internal_detectors],
        "output_detectors": [(instantiate_component(f), f["name"]) for f in raw_output_detectors],
        "safety_config": raw_config.get("safety_config", {})
    }