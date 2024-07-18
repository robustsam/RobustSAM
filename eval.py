import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import torch
import torch.nn as nn

from robust_segment_anything import SamPredictor, sam_model_registry
from robust_segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from robust_segment_anything.utils.transforms import ResizeLongestSide 

def show_boxes(coords, ax):
    x1, y1, x2, y2 = coords
    width = x2-x1
    height = y2-y1    
    bbox = patches.Rectangle((x1, y1), width, height, linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(bbox)
   
def show_points(coords, labels, ax, marker_size=500):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.8])
        
    h, w = mask.shape[-2:]    
    mask = mask.detach().cpu().numpy()   
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
parser = argparse.ArgumentParser()

parser.add_argument("--case", type=str, default="clear")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument("--data_dir", type=str, default="demo_images")
parser.add_argument("--bbox", action='store_true')
parser.add_argument("--model_size", type=str, default="l")
parser.add_argument("--checkpoint_path", type=str, default="robustsam_checkpoint_l.pth")

opt = parser.parse_args()

opt.checkpoint_path = 'robustsam_checkpoint_{}.pth'.format(opt.model_size)    
model = sam_model_registry["vit_{}".format(opt.model_size)](opt=opt, checkpoint=opt.checkpoint_path)

model = model.to(opt.gpu)
print('Succesfully loading model from {}'.format(opt.checkpoint_path))

sam_transform = ResizeLongestSide(model.image_encoder.img_size)

image_list = [os.path.join(opt.data_dir, f) for f in os.listdir(opt.data_dir)]
image_list = sorted(image_list)

box_list = [np.array([48, 38, 341, 289]), # blur
            np.array([114, 61, 386, 266]), # haze
            np.array([56, 47, 1190, 612]), # lowlight
            np.array([1, 1, 252, 298])] # rain

point_list = [np.array([[131, 233], [186, 84], [266, 54]]), # blur
              np.array([[182, 128], [152, 224], [309, 216]]), # haze
              np.array([[415, 133], [913, 463], [1030, 297]]), # lowlight
              np.array([[70, 65], [136, 172], [228, 187]])] # rain

if opt.bbox:
    input = zip(image_list, box_list)
    print('Use bounding box as prompt !')
    save_dir = os.path.join('demo_result', 'box_{}'.format(opt.model_size))

else: 
    input = zip(image_list, point_list)
    print('Use point as prompt !')
    save_dir = os.path.join('demo_result', 'point_{}'.format(opt.model_size))

os.makedirs(save_dir, exist_ok=True)
print('Result will be saved to {} ...'.format(save_dir))

for (image_path, prompt) in tqdm(input):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image_t = torch.tensor(image, dtype=torch.uint8).unsqueeze(0).to(opt.gpu)
    image_t = torch.permute(image_t, (0, 3, 1, 2))
    image_t_transformed = sam_transform.apply_image_torch(image_t.float())

    data_dict = {}
    if not opt.bbox :
        input_label = torch.Tensor(np.ones(prompt.shape[0])).to(opt.gpu)
        point_t = torch.Tensor(prompt).to(opt.gpu)
        data_dict['image'] = image_t_transformed
        data_dict['point_coords'] = sam_transform.apply_coords_torch(point_t, image_t.shape[-2:]).unsqueeze(0)
        data_dict['point_labels'] = input_label.unsqueeze(0)
        data_dict['original_size'] = image_t.shape[-2:]
    
    else:        
        box_t = torch.Tensor(prompt).unsqueeze(0).to(opt.gpu)
        data_dict['image'] = image_t_transformed
        data_dict['boxes'] = sam_transform.apply_boxes_torch(box_t, image_t.shape[-2:]).unsqueeze(0)          
        data_dict['original_size'] = image_t.shape[-2:]  

    with torch.no_grad():   
        batched_output = model.predict(opt, [data_dict], multimask_output=False, return_logits=False)    
    
    output_mask = batched_output[0]['masks']
    plt.figure(figsize=(10,10))
    plt.imshow(image[:, :, ::-1])
    
    if opt.bbox:
        show_boxes(prompt, plt.gca())
    
    else:
        show_points(prompt, np.ones(prompt.shape[0]), plt.gca(), marker_size=205)
        
    show_mask(output_mask[0][0], plt.gca())
    plt.axis('off')
    save_path = os.path.join(save_dir, image_path.split('/')[-1])
    plt.savefig(save_path, bbox_inches='tight')

print('Finish inferencing...')

