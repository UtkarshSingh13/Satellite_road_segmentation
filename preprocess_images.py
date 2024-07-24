# import necessary libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
print("hello")
train_x = sorted(glob.glob(r"C:\Users\singh\Downloads\train\*_sat.jpg"))
train_y = sorted(glob.glob(r"C:\Users\singh\Downloads\train\*_mask.png"))
print(len(train_x), len(train_y))
print(train_x[2])
img = Image.open(train_y[2])
np_img = np.array(img)
print(img.size)
print(np_img.shape)
print(train_y[2])

def data_loader(folder_list):
    
    image_dataset = []
    for i,images in enumerate(folder_list):
        image = cv2.imread(images, 1)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Binarize the image for maskom;y
        _, binary_image = cv2.threshold(gray_image, 128, 1, cv2.THRESH_BINARY)
        binary_image = np.expand_dims(binary_image, axis=-1)

        image_dataset.append(binary_image)
    print(image_dataset[600])
    return image_dataset
image_dataset = data_loader(train_y) # real images...
image_dataset = np.array(image_dataset)
np.save('train_yy.npy', image_dataset)
print(image_dataset.shape)
print("done")