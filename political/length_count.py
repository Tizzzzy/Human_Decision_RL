import json
import statistics

# Initialize lists to store the lengths
character_counts = []
word_counts = []

# Open and read the JSONL file
with open('speeches_sampled_explanation_claude.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # Parse the JSON object on each line
        data = json.loads(line)
        
        # Extract the 'text' string
        text_content = data['text']
        
        # Calculate character length
        char_length = len(text_content)
        character_counts.append(char_length)
        
        # Calculate word length (splitting by whitespace)
        word_length = len(text_content.split())
        word_counts.append(word_length)

# Calculate and print summary statistics
with open('length_analysis_results.txt', 'w', encoding='utf-8') as output_file:
    for count in word_counts:
        output_file.write(f"{count}\n")
        
print("--- Text Length Analysis ---")
print(f"Total entries analyzed: {len(character_counts)}")

print("\n--- Character Count ---")
print(f"Average: {statistics.mean(character_counts):.2f}")
print(f"Minimum: {min(character_counts)}")
print(f"Maximum: {max(character_counts)}")

print("\n--- Word Count ---")
print(f"Average: {statistics.mean(word_counts):.2f}")
print(f"Minimum: {min(word_counts)}")
print(f"Maximum: {max(word_counts)}")