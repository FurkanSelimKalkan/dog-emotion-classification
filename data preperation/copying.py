import os
import shutil
import random

# Set the path to the main folder
main_folder = '/images_downloaded'

# Set the names of the test, validation, and train folders
test_folder = 'test'
val_folder = 'val'
train_folder = 'train'

# Set the name of the new folder for the 10% subset
subset_folder = 'subset'

# Create the subset folder
subset_path = os.path.join(main_folder, subset_folder)
os.makedirs(subset_path, exist_ok=True)

# Iterate over the main folder's subfolders
for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)

    # Skip the subset folder and any non-directory files
    if folder_name == subset_folder or not os.path.isdir(folder_path):
        continue

    # Create the corresponding folders in the subset folder
    subset_folder_path = os.path.join(subset_path, folder_name)
    os.makedirs(subset_folder_path, exist_ok=True)

    # Get the files in the current folder
    files = os.listdir(folder_path)

    # Shuffle the files randomly
    random.shuffle(files)

    # Calculate the number of files to copy for the subset (10% of the total)
    num_files = len(files)
    num_subset_files = int(num_files * 0.1)

    # Copy the subset of files to the subset folder
    subset_files = files[:num_subset_files]
    for file_name in subset_files:
        src_path = os.path.join(folder_path, file_name)
        dst_path = os.path.join(subset_folder_path, file_name)
        shutil.copy(src_path, dst_path)

print("Subset creation completed.")
