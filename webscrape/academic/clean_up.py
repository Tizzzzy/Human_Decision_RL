import os
from pathlib import Path

def sync_academic_rewrite_folder(target_dir, base_dir, claude_dir, gemini_dir):
    # 1. Helper to get all relative file paths from a directory
    def get_file_set(directory):
        path = Path(directory)
        # We store the relative path to compare across different directory roots
        return {str(p.relative_to(path)) for p in path.rglob('*.txt')}

    print("Scanning directories for file overlap...")
    
    # Get sets of files for the three explanation folders
    base_files = get_file_set(base_dir)
    claude_files = get_file_set(claude_dir)
    gemini_files = get_file_set(gemini_dir)

    # 2. Find the intersection (Files that exist in ALL THREE)
    valid_files = base_files.intersection(claude_files).intersection(gemini_files)
    
    print(f"Files found in all explanation folders: {len(valid_files)}")

    # 3. Iterate through the Academic_rewrite folder and delete orphans
    target_path = Path(target_dir)
    deleted_count = 0
    
    print(f"Cleaning up: {target_path}...")

    # We use rglob to find all .txt files in the target folder
    for file_path in target_path.rglob('*.txt'):
        rel_path = str(file_path.relative_to(target_path))
        
        if rel_path not in valid_files:
            try:
                os.remove(file_path)
                print(f"Deleted: {rel_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {rel_path}: {e}")

    print("\n--- Sync Complete ---")
    print(f"Files kept: {len(valid_files)}")
    print(f"Files deleted: {deleted_count}")

# --- Configuration ---
category = "Academic"
TARGET_DIR = f'{category}' # The folder you want to clean
BASE_DIR = f'{category}_explanation'
CLAUDE_DIR = f'{category}_explanation_claude'
GEMINI_DIR = f'{category}_explanation_gemini'

if __name__ == "__main__":
    sync_academic_rewrite_folder(TARGET_DIR, BASE_DIR, CLAUDE_DIR, GEMINI_DIR)