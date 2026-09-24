import os

dir1 = './Poetry_rewrite'
dir2 = './Poetry'

# Get lists of only files (excluding directories)
files1 = set(f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f)))
files2 = set(f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f)))

# Find common filenames
common_files = files1.intersection(files2)
print(f"Number of common files: {len(common_files)}")
