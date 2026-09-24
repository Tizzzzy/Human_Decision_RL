import os
import matplotlib.pyplot as plt

# 1. Define your directories
INPUT_DIR = "News"

def process_academic_directory():
    total_words = 0
    file_count = 0
    word_counts = []  # List to store lengths for plotting

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".txt"):
                input_file_path = os.path.join(root, file)
                
                try:
                    with open(input_file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                        
                    if not text_content:
                        continue
                    
                    # Calculate word count for this specific file
                    words = text_content.split()
                    current_count = len(words)
                    
                    # Update metrics
                    word_counts.append(current_count)
                    total_words += current_count
                    file_count += 1
                        
                except Exception as e:
                    print(f"  -> Error processing {input_file_path}: {e}")

    if file_count > 0:
        average_words = total_words / file_count
        print("\n--- Statistics ---")
        print(f"Total Files Processed: {file_count}")
        print(f"Total Word Count: {total_words}")
        print(f"min Words in a File: {min(word_counts)}")
        print(f"max Words in a File: {max(word_counts)}")
        print(f"Average Words per File: {average_words:.2f}")
        
        # --- Plotting Section ---
        plot_distribution(word_counts)
    else:
        print("\nNo valid text files found.")

def plot_distribution(data):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of Document Lengths (Word Count)')
    plt.xlabel('Number of Words')
    plt.ylabel('Number of Files')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    print("\nDisplaying plot...")
    plt.savefig(f"{INPUT_DIR}_length_distribution.png")  # Save the plot as an image file

if __name__ == "__main__":
    process_academic_directory()