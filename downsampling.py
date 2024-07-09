import os
import cv2

def create_low_res_images(input_dir, output_dir, scale=0.5):
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
                    
                    low_res_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    
                    output_path = os.path.join(output_city_dir, file)
                    cv2.imwrite(output_path, low_res_img)
                    print(f"Processed {os.path.normpath(output_path)}")

input_dir = "reduced_data/leftImg8bit_trainvaltest"
output_dir = "reduced_data/leftImg8bit_low_res"

create_low_res_images(input_dir, output_dir)
