import json
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from tqdm import tqdm

# 1. Load the compiled dataset from the previous step
input_file = "/projects/p32143/RL_human_decision/simulator/data/prompts_dataset.json"
with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 2. Load Model and Processor
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.6-27B")
model = AutoModelForMultimodalLM.from_pretrained(
    "Qwen/Qwen3.6-27B", 
    device_map="auto", 
    cache_dir="/projects/p32143/cache/huggingface/qwen36_27b"
)
model.eval()

# Use JSON Lines (.jsonl) to write each record sequentially. 
output_filename = "/projects/p32143/RL_human_decision/simulator/data/extracted_representations.jsonl"

# 3. Process each prompt
print(f"Beginning extraction for {len(dataset)} items...")
with open(output_filename, "w", encoding="utf-8") as out_f:
    for item in tqdm(dataset, desc="Processing Prompts"):
        text_id = item["text_id"]
        prob_ai = item["probability_label_1"]
        prompt_text = item["prompt"]
        
        # Pulling these to uniquely identify duplicate text_ids
        label_truth = item["label_truth"]
        explanation_source = item["explanation_source"]
        all_guesses = item["raw_guesses_numeric"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            # Perform a single forward pass instead of generating new tokens
            outputs = model(**inputs, output_hidden_states=True)

        # 4. Extract Final Layer Residual Stream
        # outputs.hidden_states is a tuple of layers. [-1] gets the final layer.
        # Shape: (batch_size, sequence_length, hidden_dimension)
        final_layer_states = outputs.hidden_states[-1]
        
        # Extract the representation of the very last token in the input prompt
        # [0, -1, :] = Batch 0, Last Token, All Dimensions
        last_token_final_layer = final_layer_states[0, -1, :]

        # 5. Format and Save
        output_data = {
            "text_id": text_id,
            "label_truth": label_truth,
            "explanation_source": explanation_source,
            "probability_label_1": prob_ai,
            "raw_guesses_numeric": all_guesses,
            "representation_shape": list(last_token_final_layer.shape),
            "last_token_residual_stream": last_token_final_layer.cpu().float().tolist()
        }

        # Write to JSONL
        out_f.write(json.dumps(output_data) + "\n")
        
        # Clear cache to prevent memory buildup during long loops
        torch.cuda.empty_cache()

print(f"Representations successfully extracted and saved to {output_filename}")