from typing import Any, Collection

# disapprovals is for example {"class_name": LPDetector, "approved": False}
# if any disapprove => disapprove simple rule
def classify_any_or_rule(input_disapprovals: Collection[dict[str, Any]],internal_disapprovals: Collection[list[dict[str, Any]]], output_disapprovals: Collection[list[dict[str, Any]]]) -> bool:
    """ Applies an OR decision across input, internal, and output safety evaluations.

    Args:
        input_disapprovals: Evaluation dicts from the input classifier
        internal_disapprovals: Evaluation dicts from the internal probe
        output_disapprovals: Evaluation dicts from output classifier

    Returns:
        bool: 'True' if any classifiers disapproved, 'False' otherwise
    """

    overall_disapproval = (
        any(v["disapproved"] for v in input_disapprovals) or 
        any(v["disapproved"] for v in internal_disapprovals) or 
        any(v["disapproved"] for v in output_disapprovals)
    )
    return overall_disapproval


def classify_input_and_internal_or_output_rule(input_disapprovals,internal_disapprovals, output_disapprovals):
    """ Applies a combined decision rule: input AND eternal OR output."""
    overall_disapproval = (
        ( any(v["disapproved"] for v in input_disapprovals) and 
        any(v["disapproved"] for v in internal_disapprovals) ) or 
        any(v["disapproved"] for v in output_disapprovals)
    )
    return overall_disapproval

