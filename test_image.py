import os

image_folder = '/Users/dengjiaxin/project/mtwi_to_detectron2/MTWI2018/mtwi_train_image/'

image_files = [f for f in os.listdir(image_folder)]
for image in image_files:
    if not image.endswith('.jpg'):
        raise(image)
    print(image)