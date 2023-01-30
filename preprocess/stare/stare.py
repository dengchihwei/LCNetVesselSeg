# -*- coding = utf-8 -*-
# @File Name : stare
# @Date : 2022/10/15 17:34
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import numpy as np
from PIL import Image
from skimage.morphology import remove_small_objects


# generate masks for STARE images
def gen_mask_images(data_path):
    image_path = os.path.join(data_path, 'images')
    os.makedirs(os.path.join(data_path, 'masks'), exist_ok=True)
    for img_file in sorted(os.listdir(image_path)):
        if img_file[0] != 'i':
            continue
        image = Image.open(os.path.join(image_path, img_file))
        image = image.convert('L')
        image = image.point(lambda p: 1 if p > 40 else 0)
        image = np.array(image) > 0
        image = remove_small_objects(image, min_size=20)
        image = (image * 255).astype(np.uint8)
        mask = Image.fromarray(image)
        mask.save(os.path.join(data_path, 'masks', img_file))


gen_mask_images('/Users/zhiweideng/Desktop/NICR/VesselAnalysis/EyeVessel/STARE')
