import os
from uuid import uuid4  
from tqdm.auto import tqdm


def reverse_files(folder_path):
    # Get all files in the directory, excluding subdirectories
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Sort the files lexicographically
    sorted_files = sorted(files)
    
    # Determine the number of pairs to swap
    n = len(sorted_files)
    for i in tqdm(range(n // 2), desc=f"Swapping files in {folder_path}"):
        # Get the i-th file from the start and end
        file1 = sorted_files[i]
        file2 = sorted_files[n - i - 1]
        
        # Full paths for the files
        path1 = os.path.join(folder_path, file1)
        path2 = os.path.join(folder_path, file2)
        
        # Read the contents of both files
        with open(path1, 'rb') as f:
            content1 = f.read()
        with open(path2, 'rb') as f:
            content2 = f.read()
        
        # Write the swapped contents back to the files
        with open(path1, 'wb') as f:
            f.write(content2)
        with open(path2, 'wb') as f:
            f.write(content1)
        


def make_tmp_folder():
    folder = f'/tmp/mb/sam2gui/{uuid4()}'
    os.makedirs(folder, exist_ok=True)
    return folder