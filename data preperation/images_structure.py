import os
import shutil
import random

# Set the path to your raw image folder
raw_img_dir = "/images_downloaded"

# Set the path to the target dataset directory
dataset_dir = "/to_clean_images"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

# Create the dataset directory and its subdirectories
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the split ratios for train, validation, and test sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Get the list of animal class folders
animal_classes = os.listdir(raw_img_dir)

# Iterate over each animal class folder
for animal_class in animal_classes:
    class_dir = os.path.join(raw_img_dir, animal_class)
    images = os.listdir(class_dir)
    random.shuffle(images)

    # Calculate the number of images for each split
    num_images = len(images)
    num_train = int(train_ratio * num_images)
    num_val = int(val_ratio * num_images)
    num_test = num_images - num_train - num_val

    # Split the images into train, validation, and test sets
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    # Create class directories in train, validation, and test sets
    train_class_dir = os.path.join(train_dir, animal_class)
    val_class_dir = os.path.join(val_dir, animal_class)
    test_class_dir = os.path.join(test_dir, animal_class)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Move images to the corresponding directories
    for image in train_images:
        src_path = os.path.join(class_dir, image)
        dst_path = os.path.join(train_class_dir, image)
        shutil.copy(src_path, dst_path)

    for image in val_images:
        src_path = os.path.join(class_dir, image)
        dst_path = os.path.join(val_class_dir, image)
        shutil.copy(src_path, dst_path)

    for image in test_images:
        src_path = os.path.join(class_dir, image)
        dst_path = os.path.join(test_class_dir, image)
        shutil.copy(src_path, dst_path)

print("Dataset folder structure created successfully.")
