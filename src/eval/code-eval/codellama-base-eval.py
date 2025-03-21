# This code was adapted from GitHub developer 'abacaj' (https://github.com/abacaj/code-eval/)

from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from core import filter_code, run_eval, fix_indents
import os
import torch
from datasets import load_dataset

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""

@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completions]

if __name__ == "__main__":
    # Load test cases from Hugging Face dataset
    dataset = load_dataset("hababou/code-chat-assistant-v1", split="test")

    # adjust for n = 10 etc
    num_samples_per_task = 10
    out_path = "results/codellama/eval.jsonl"
    os.makedirs("results/codellama", exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(
        "meta-llama/CodeLlama-7b-Instruct-hf",
    )

    model = torch.compile(
        LlamaForCausalLM.from_pretrained(
            "meta-llama/CodeLlama-7b-Instruct-hf",
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .to("cuda")
    )

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
        dataset
    )