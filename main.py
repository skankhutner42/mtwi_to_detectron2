import json
import os

def convert_to_detectron2_format(txt_path, image_folder):
    data = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            x1, y1, x2, y2, x3, y3, x4, y4, text = line[:9]
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, [x1, y1, x2, y2, x3, y3, x4, y4])
            annotation = {
                "bbox": [min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), max(y1, y2, y3, y4)],
                "category_id": 0,  # Assuming all text belongs to the same category
                "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]],
                "text": text
            }
            data.append(annotation)

    image_name = os.path.basename(txt_path).split('.')[0] + ".jpg"
    image_path = os.path.join(image_folder, image_name)
    image_info = {
        "file_name": image_path,
        "height": 0,  # You need to fill in the correct height and width here
        "width": 0,
        "image_id": 0  # You can assign a unique ID for each image
    }

    return {
        "annotations": data,
        "images": [image_info],
        "categories": [{"id": 0, "name": "text"}]  # Assuming all text belongs to the same category
    }

# Output JSON file
output_json = "detectron2_format.json"

# Label folder and image folder
label_folder = '/Users/dengjiaxin/project/mtwi_to_detectron2/MTWI2018/mtwi_train_label/'
image_folder = '/Users/dengjiaxin/project/mtwi_to_detectron2/MTWI2018/mtwi_train_image/'
image_test = '/Users/dengjiaxin/project/mtwi_to_detectron2/MTWI2018/image_test'
# List to store all data
all_data = []

# Iterate through each .txt file
for txt_file in os.listdir(label_folder):
    if txt_file.endswith('.txt'):
        print(txt_file)
        txt_path = os.path.join(label_folder, txt_file)
        data = convert_to_detectron2_format(txt_path, image_folder)
        all_data.append(data)

# Write all data to JSON file
with open(output_json, 'w') as outfile:
    json.dump(all_data, outfile, ensure_ascii=False)
