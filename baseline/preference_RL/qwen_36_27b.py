import argparse
import json
import os
import time
from typing import Dict, Iterable, List

from PIL import Image
from vllm import LLM, SamplingParams

MODEL_PATH = "/projects/p32143/cache/qwen36_27b" # Path to your Qwen model
MODEL_ID_NAME = "Qwen/Qwen3.6-27B" # Used for saving the output folder

def format_with_chat_template(llm: LLM, content: list, enable_thinking: bool = False) -> str:
    # content is now a list of dictionaries (images + text)
    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Some tokenizer/vLLM versions pass Qwen chat-template kwargs this way.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max_model_len", type=int, default=262144)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--disable_log_stats", action="store_true", default=True)

    ap.add_argument("--batch_size", type=int, default=1) # Lower this if you get CPU RAM Out-Of-Memory errors loading many images
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--top_p", type=float, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--use_chat_template", action="store_true", default=True)
    ap.add_argument("--enable_thinking", action="store_true", help="Enable Qwen thinking mode in chat template.")
    args = ap.parse_args()

    print("🚀 Initializing vLLM Engine (This happens only once)...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # enforce_eager=args.enforce_eager,
        enforce_eager=True,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=args.disable_log_stats,
        generation_config="vllm",
        limit_mm_per_prompt={
            "image": 0, # Increased to allow up to 100 PPT slides per prompt
            "video": 0,
            "audio": 0,
        },
    )

    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed
    sampling = SamplingParams(**sampling_kwargs)

    start_time = time.time()
    # --- TEXT PROMPT ---
    prompt_text = """Who is president of the Japen? Please answer in one sentence."""

    content = [{"type": "text", "text": prompt_text}]

    # Format the prompt string via the tokenizer
    if args.use_chat_template:
        final_prompt_str = format_with_chat_template(llm, content, args.enable_thinking)
    else:
        final_prompt_str = prompt_text

    try:
        # vLLM naturally accepts a list of dictionaries for offline multimodal batched generation
        output = llm.generate(final_prompt_str, sampling)
        print(f"📝  Generated output: {output[0].outputs[0].text.strip()}")
        print(f"✅ Batch generation completed in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"❌ Error during batch generation: {e}")
        
if __name__ == "__main__":
    main()