import json
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

# 1. Load Model and Processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForMultimodalLM.from_pretrained(
    "Qwen/Qwen3.6-27B", 
    device_map="auto", 
    cache_dir="/projects/p32143/cache/huggingface/qwen36_27b"
)

# 2. Prepare Inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What the president of the United States is?"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# 3. Generate Output and Extract Hidden States
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512,
        return_dict_in_generate=True,
        output_hidden_states=True
    )

# 4. Decode Generated Text
generated_ids = outputs.sequences[0][inputs["input_ids"].shape[-1]:]
generated_text = processor.decode(generated_ids, skip_special_tokens=True)
print(f"Generated Text:\n{generated_text}\n")
