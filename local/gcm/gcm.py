#!/usr/bin/env python3
import sys
import argparse
from typing import Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedModel


def load_model(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model


def generate_commit_message(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    diff: str,
    max_new_tokens: int = 32
) -> str:
    prompt: str = (
        "Write a single concise Git commit message in imperative mood, "
        "no more than one line, summarizing the following diff:\n\n"
        f"{diff}\n\nCommit message:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    text: str = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the commit message part
    if "Commit message:" in text:
        text = text.split("Commit message:")[-1].strip()

    return text.split("\n")[0].strip()  # enforce one line


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate a Git commit message from diff"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="codellama/CodeLlama-7b-hf",
        help="Hugging Face model to use"
    )
    args: argparse.Namespace = parser.parse_args()

    diff: str = sys.stdin.read().strip()
    if not diff:
        print("No diff provided on stdin.", file=sys.stderr)
        sys.exit(1)

    tokenizer, model = load_model(args.model)
    message: str = generate_commit_message(tokenizer, model, diff)
    print(message)


if __name__ == "__main__":
    main()
