from pathlib import Path
from typing import  List
from safeguard_llm.safety_harness import GenerationSafetyResult
import json
from dataclasses import asdict
def save_results_as_json(results: List[GenerationSafetyResult], output_path: str | Path) -> None:

        out_path = Path(output_path)
        serialized_results = [asdict(res) for res in results]

        with open(out_path, "w") as f:
            json.dump(serialized_results, f, indent=4)        