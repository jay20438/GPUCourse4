import os
import cv2

def resize_and_save_images(input_dir, output_dir, size):
    for split in ['train', 'test', 'val']:
        input_split_dir = os.path.join(input_dir, split)
        output_split_dir = os.path.join(output_dir, split)

        if not os.path.exists(output_split_dir):
            os.makedirs(output_split_dir)

        for city in os.listdir(input_split_dir):
            input_city_dir = os.path.join(input_split_dir, city)
            output_city_dir = os.path.join(output_split_dir, city)

            if not os.path.exists(output_city_dir):
                os.makedirs(output_city_dir)

            for file in os.listdir(input_city_dir):
                if file.endswith(".png"):
                    img_path = os.path.join(input_city_dir, file)
                    img = cv2.imread(img_path)

                    # Resize image
                    resized_img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

                    output_path = os.path.join(output_city_dir, file)
                    cv2.imwrite(output_path, resized_img)
                    print(f"Processed {os.path.normpath(output_path)}")

# Define paths and sizes
input_dir_high_res = "reduced_data/leftImg8bit_trainvaltest"
output_dir_high_res = "reduced_data/resized_high_res"
high_res_size = (256, 256)

input_dir_low_res = "reduced_data/leftImg8bit_trainvaltest"
output_dir_low_res = "reduced_data/resized_low_res"
low_res_size = (128, 128)

# Resize and save high-resolution images
resize_and_save_images(input_dir_high_res, output_dir_high_res, high_res_size)

# Resize and save low-resolution images
resize_and_save_images(input_dir_low_res, output_dir_low_res, low_res_size)
