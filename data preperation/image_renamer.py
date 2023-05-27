import os

folder_path = "/images_downloaded/sad"

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Sort the image files
image_files.sort()

# Rename the image files
for i, image_file in enumerate(image_files):
    old_path = os.path.join(folder_path, image_file)
    new_name = str(i+1) + ".jpg"  # Assuming the images are in JPEG format, change the extension accordingly if needed
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)

print("Image files renamed successfully!")
