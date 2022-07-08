import torchvision as tv
from PIL import Image
import cv2
import numpy as np

img_path = r'../train_images/291_1.jpg'

pil_img = Image.open(img_path)

while True:
    # tv_affine = tv.transforms.RandomAffine(degrees=10, translate=(0.1,0.2), scale=(0.9,1.1),shear=0)
    tv_affine = tv.transforms.RandomPerspective(distortion_scale=0.1, p=0.1)
    img = tv_affine(pil_img)

    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    cv2.imshow('',img)
    cv2.waitKey()