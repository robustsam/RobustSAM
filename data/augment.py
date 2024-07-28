import cv2
from glob import glob
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torchvision.transforms as T
import albumentations as A
import imgaug.augmenters as iaa
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm

def transform(case, path):        
    ori_image = cv2.imread(path)
    clear_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    
    degraded_dict = {'snow': snow, 'fog': fog, 'rain': rain,
                     'gauss_noise': gauss_noise, 'ISO_noise': ISO_noise, 'impulse_noise': impulse_noise,
                     'resampling_blur': resampling_blur, 'motion_blur': motion_blur, 'zoom_blur': zoom_blur,
                     'color_jitter': color_jitter, 'compression': compression, 'elastic_transform': elastic_transform,   
                     'frosted_glass_blur': frosted_glass_blur, 'brightness': brightness, 'contrast': contrast,
                     }
    
    image = degraded_dict[case](image)    
   
    return image


def snow(image):
    transform = A.Compose(
        [A.RandomSnow(brightness_coeff=1.0, snow_point_lower=0.3, snow_point_upper=0.7, p=1)],
    )

    image = transform(image=image)['image']

    aug = iaa.Snowflakes(density=0.35, flake_size=(0.6, 0.8), speed=(0.01, 0.015), angle=0)
    image = aug.augment_image(image)   
    return image

def fog(image):
    aug = iaa.Fog(seed=1234)
    image = aug.augment_image(image)   

    fog_coef_lower = 0.1
    fog_coef_upper = 0.2
    alpha_coef = 0.08
    transform = A.Compose(
        [A.RandomFog(fog_coef_lower=fog_coef_lower, fog_coef_upper=fog_coef_upper, alpha_coef=alpha_coef, p=1)],
    )    
    image = transform(image=image)
    image = image['image'] 

    return image

def rain(image):
    aug = iaa.Rain(drop_size=(0.40, 0.50), speed=(0.05, 0.1))
    image = aug.augment_image(image)
    
    return image

def gauss_noise(image):
    var = (250, 260)
    
    transform = A.Compose(
        [A.GaussNoise(var_limit=var, per_channel=True, p=1)],
    )        
    image = transform(image=image)
    image = image['image']   
    
    return image

def ISO_noise(image):  
    color_shift = (0.4, 0.5)
    intensity = (0.7, 0.8)
    
    transform = A.Compose(
        [A.ISONoise(color_shift=color_shift, intensity=intensity, p=1)],
    ) 
    image = transform(image=image)
    image = image['image']     
    
    return image

def impulse_noise(image):
    multiplier = (1.4, 1.5)
    
    transform = A.Compose(
            [A.MultiplicativeNoise(multiplier=multiplier, p=1)],
        )     
    
    image = transform(image=image)
    image = image['image'] 
    
    return image

def resampling_blur(image):
    resize_factor = 4
    ori_height, ori_width =  image.shape[0], image.shape[1]
    new_height, new_width = int(image.shape[0]/resize_factor), int(image.shape[1]/resize_factor)
    img_down = cv2.resize(image, (new_width, new_height))
    image = cv2.resize(img_down, (ori_width, ori_height))
    
    return image

def motion_blur(image):
    blur_limit = 35
    
    transform = A.Compose(
        [A.MotionBlur(blur_limit=blur_limit, p=1)],
    )    
    
    image = transform(image=image)
    image = image['image'] 
    
    return image

def zoom_blur(image):         
    max_factor = 1.33
    step_factor = (0.05, 0.07)
    
    transform = A.Compose(
        [A.ZoomBlur(max_factor=max_factor, step_factor=step_factor, p=1)],
    )    
    
    image = transform(image=image)
    image = image['image'] 
    
    return image

def color_jitter(image):
    brightness = 0.7
    contrast = 0.3
    saturation = 0.7
    hue = 0.7
    
    transform = A.Compose(
        [A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=1)],
    )        
    
    image = transform(image=image)
    image = image['image']  
    
    return image

def compression(image):      
    quality_lower = 5
    quality_upper = 10
    
    transform = A.Compose(
        [A.ImageCompression(quality_lower=quality_lower, quality_upper=quality_upper, p=1)],
    )    
    
    image = transform(image=image)
    image = image['image']  
    
    return image

def elastic_transform(image):          
    alpha = 100
    sigma = 10
    alpha_affine = 5
    
    transform = A.Compose(
        [A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, p=1)],
    )    
    image = transform(image=image)
    image = image['image'] 
    
    return image

def frosted_glass_blur(image):    
    max_displacement = 3
    
    height, width, _ = image.shape
    augmented_image = np.copy(image)

    for y in range(height):
        for x in range(width):
            displacement_x = np.random.randint(-max_displacement, max_displacement + 1)
            displacement_y = np.random.randint(-max_displacement, max_displacement + 1)

            target_x = max(0, min(width - 1, x + displacement_x))
            target_y = max(0, min(height - 1, y + displacement_y))

            augmented_image[y, x] = image[target_y, target_x]

    return augmented_image

def brightness(image):
    factor = random.uniform(0.6, 0.8)
    augmented_image = np.clip(image * factor, 0, 255).astype(np.uint8)    

    return augmented_image

def contrast(image):    
    factor = random.uniform(1.6, 1.7)
    image = image.astype(np.float32)

    augmented_image = (image - 128) * factor + 128

    augmented_image = np.clip(augmented_image, 0, 255)
    augmented_image = augmented_image.astype(np.uint8)
    
    return augmented_image

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="all_data/train")
parser.add_argument("--case", type=str, default=None)

opt = parser.parse_args()

image_path = os.path.join(opt.data_dir, 'clear')
image_list = sorted(glob(os.path.join(image_path, '*jpg')))
print('Number of images: {}'.format(len(image_list)))

if opt.case is None:
    degraded_list = ['snow', 'fog', 'rain', 'gauss_noise', 'ISO_noise', 'impulse_noise', 'resampling_blur', 
                    'motion_blur', 'zoom_blur', 'color_jitter', 'compression', 'elastic_transform',
                    'frosted_glass_blur', 'brightness', 'contrast' ]

else: 
    degraded_list = [case]
    print('Processing {} only!'.format(case))


for j, degraded_type in enumerate(degraded_list):
    case = degraded_type
    save_folder = os.path.join(opt.data_dir, case)
    os.makedirs(save_folder, exist_ok=True)
    print('Generating {} images. Degraded images will be saved to {} ...'.format(degraded_type, save_folder))
    
    for i, path in enumerate(tqdm(image_list)):  
        augmented_image = transform(case, path)
        fname = path.split('/')[-1]
        save_path = os.path.join(save_folder, fname)
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, augmented_image)



        










    
    