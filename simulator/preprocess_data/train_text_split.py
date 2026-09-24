import json
from sklearn.model_selection import train_test_split

input_file = "/projects/p32143/RL_human_decision/simulator/data/extracted_representations.jsonl"
train_file = "/projects/p32143/RL_human_decision/simulator/data/train_set.jsonl"
test_file  = "/projects/p32143/RL_human_decision/simulator/data/test_set.jsonl"

# 1. Load the data
data = []
labels = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        data.append(item)
        labels.append(item["label_truth"])

# 2. Split the data 
# stratify=labels ensures the train/test sets have the same distribution of true labels
train_data, test_data = train_test_split(
    data, 
    test_size=0.1, 
    random_state=42, 
    stratify=labels 
)

# 3. Save files
with open(train_file, 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open(test_file, 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')
        
print(f"Train size: {len(train_data)} | Test size: {len(test_data)}")