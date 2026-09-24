from transformers import AutoProcessor, AutoModelForMultimodalLM
from time import time

CACHE_DIR = "/projects/p32143/cache/huggingface"
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B", cache_dir=CACHE_DIR)
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-9B", device_map="auto", cache_dir=CACHE_DIR)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Who is the president of the United States?"}
        ]
    },
]

start_time = time()

inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
end_time = time()
print(f"Generation time: {end_time - start_time} seconds")