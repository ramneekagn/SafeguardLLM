from transformers import AutoModelForCausalLM, AutoTokenizer
def get_model_tokenizer() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """ Loads and intiailizes a Qwen3-0.6B model and its tokenizer

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenize]:
            - The loaded Qwen3-0.6B model
            - The corresponding tokenizer

    """

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer