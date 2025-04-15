def format_example(example: dict) -> tuple[str, str]:
    """
    Custom function that returns (prompt_text, target_text) from a given example.
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output_text = example["output"]

    # Compose the prompt using your special tokens
    prompt = (
        "<|system|>\n"
        "You are a helpful assistant.\n"
        "<|user|>\n"
        f"<|begin_instruction|>{instruction}{'<|end_instruction|>'}"
    )

    # If there's some additional input field, you can weave it in:
    # e.g. if input_text is non-empty, you can put it after the instruction
    if input_text.strip():
        prompt += f"\n<|begin_context|>{input_text}<|end_context|>"

    # The “target” or “label” that we want the model to produce:
    target = f"<|assistant|>\n<|begin_response|>{output_text}<|end_response|>"

    return prompt, target
