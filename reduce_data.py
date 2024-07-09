import os
import shutil

def copy_first_third_images(source_dir, target_dir):
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Traverse through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Get the current subfolder name relative to source_dir
        relative_folder = os.path.relpath(root, source_dir)
        target_folder = os.path.join(target_dir, relative_folder)

        # Create target subfolder if it doesn't exist
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Determine number of images to copy (1/3 of total)
        num_images = len(files)
        num_to_copy = num_images // 3

        # Copy the first 1/3 images to the target subfolder
        for i in range(num_to_copy):
            src_file = os.path.join(root, files[i])
            dst_file = os.path.join(target_folder, files[i])
            shutil.copyfile(src_file, dst_file)
            print(f"Copied: {os.path.normpath(src_file)} to {os.path.normpath(dst_file)}")

# Paths to the Cityscapes dataset folders
gtFine_dir = 'gtFine_trainvaltest/gtFine'
leftImg8bit_dir = 'leftImg8bit_trainvaltest/leftImg8bit'

# Target directories to store selected images
output_dir = 'reduced_data'  # Adjust as per your preference

# Process gtFine_trainvaltest folder
copy_first_third_images(gtFine_dir, os.path.join(output_dir, 'gtFine_trainvaltest'))

# Process leftImg8bit_trainvaltest folder
copy_first_third_images(leftImg8bit_dir, os.path.join(output_dir, 'leftImg8bit_trainvaltest'))




# import os
# import shutil

# def copy_first_quarter_images(source_dir, target_dir):
#     # Create target directory if it doesn't exist
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)

#     # Traverse through the source directory
#     for root, dirs, files in os.walk(source_dir):
#         # Get the current subfolder name relative to source_dir
#         relative_folder = os.path.relpath(root, source_dir)
#         target_folder = os.path.join(target_dir, relative_folder)

#         # Create target subfolder if it doesn't exist
#         if not os.path.exists(target_folder):
#             os.makedirs(target_folder)

#         # Determine number of images to copy (1/4 of total)
#         num_images = len(files)
#         num_to_copy = num_images // 4

#         # Copy the first 1/4 images to the target subfolder
#         for i in range(num_to_copy):
#             src_file = os.path.join(root, files[i])
#             dst_file = os.path.join(target_folder, files[i])
#             shutil.copyfile(src_file, dst_file)
#             print(f"Copied: {os.path.normpath(src_file)} to {os.path.normpath(dst_file)}")

# # Paths to the Cityscapes dataset folders
# gtFine_dir = 'gtFine_trainvaltest/gtFine'
# leftImg8bit_dir = 'leftImg8bit_trainvaltest/leftImg8bit'

# # Target directories to store selected images
# output_dir = 'filtered_data'  # Adjust as per your preference

# # Process gtFine_trainvaltest folder
# copy_first_quarter_images(gtFine_dir, os.path.join(output_dir, 'gtFine_trainvaltest'))

# # Process leftImg8bit_trainvaltest folder
# copy_first_quarter_images(leftImg8bit_dir, os.path.join(output_dir, 'leftImg8bit_trainvaltest'))

