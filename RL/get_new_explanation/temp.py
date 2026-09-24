import json

input_filename = 'rlvr_explanations.json'
output_filename = 'inference_rlvr.json'

with open(input_filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    if 'new_explanation' in item:
        # Remove the old explanation and replace it with the new one
        item.pop('explanation', None) 
        item['explanation'] = item.pop('new_explanation')

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)