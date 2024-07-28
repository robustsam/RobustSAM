import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from loss import *

criterion_focal, criterion_dice, criterion_token, criterion_embeddings = FocalLoss(gamma=2, alpha=0.25), DiceLoss(smooth=1e-6, reduction='mean'), nn.MSELoss(), nn.MSELoss()
weight_focal, weight_dice, weight_token, weight_embedding = 20, 1, 2, 100

def train(opt, epoch, optimizer, train_loader, sam_transform, model):
    model.train()
    epoch_loss = 0
    epoch_count = 0
    
    pbar = tqdm(train_loader)
    for batch in pbar:
        clear_im, degraded_im, clear_fname, all_gt_masks, points, labels = batch 
        
        clear_im = clear_im.cuda()
        degraded_im = degraded_im.cuda()
        gt_mask = all_gt_masks.cuda()
        points = points.cuda()
        labels = labels.cuda()
        
        original_im = torch.permute(clear_im, (0, 3, 1, 2)) 
        degraded_im = torch.permute(degraded_im, (0, 3, 1, 2)) 

        all_im = torch.cat((original_im, degraded_im), dim=0) 
        all_im_transformed = sam_transform.apply_image_torch(all_im.float())  
        
        all_points = torch.cat((points, points), dim=0)
        all_labels = torch.cat((labels, labels), dim=0)         

        batched_input = []
        for i in range(all_im_transformed.shape[0]):
            data_dict = {}
            data_dict['image'] = all_im_transformed[i]
            data_dict['point_coords'] = sam_transform.apply_coords_torch(all_points[i], all_im.shape[-2:]).unsqueeze(0)
            data_dict['point_labels'] = all_labels[i].unsqueeze(0)
            data_dict['original_size'] = all_im.shape[-2:]                
            batched_input.append(data_dict)
        
        batched_output = model(opt, batched_input, multimask_output=False, return_logits=True) 

        degraded_index = int(0.5 * len(batched_input))         

        clear_masks = batched_output[0]['masks'] 
        clear_embeddings = batched_output[0]['robust_embeddings']
        clear_tokens = batched_output[0]['robust_token']
        for i in range(1, degraded_index):
            clear_masks = torch.cat((clear_masks, batched_output[i]['masks']), dim=0)
            clear_embeddings = torch.cat((clear_embeddings, batched_output[i]['robust_embeddings']), dim=0)
            clear_tokens = torch.cat((clear_tokens, batched_output[i]['robust_token']), dim=0)  

        # get model output of degraded images
        degraded_masks = batched_output[degraded_index]['masks']
        degraded_embeddings = batched_output[degraded_index]['robust_embeddings']
        degraded_tokens = batched_output[degraded_index]['robust_token']           
        for i in range(degraded_index+1, len(batched_output)):
            degraded_masks = torch.cat((degraded_masks, batched_output[i]['masks']), dim=0)
            degraded_embeddings = torch.cat((degraded_embeddings, batched_output[i]['robust_embeddings']), dim=0)
            degraded_tokens = torch.cat((degraded_tokens, batched_output[i]['robust_token']), dim=0)

        optimizer.zero_grad() 

        gt_mask = gt_mask.float().unsqueeze(1)       
        dice_loss_clear = criterion_dice(degraded_masks, clear_masks.float())          
        focal_loss_clear = criterion_focal(degraded_masks, clear_masks.float())  
        mask_loss_clear = weight_focal*focal_loss_clear + weight_dice*dice_loss_clear
        
        dice_loss_gt = criterion_dice(degraded_masks, gt_mask)          
        focal_loss_gt = criterion_focal(degraded_masks, gt_mask)       
        mask_loss_gt = weight_focal*focal_loss_gt + weight_dice*dice_loss_gt
        
        mask_loss = mask_loss_gt        
        token_loss = criterion_token(degraded_tokens, clear_tokens)*weight_token
        embeddings_loss = criterion_embeddings(degraded_embeddings, clear_embeddings)*weight_embedding
        
        total_loss = mask_loss + token_loss + embeddings_loss
        total_loss.backward() 
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of {name}, setting to zero.")
                    param.grad[torch.isnan(param.grad)] = 0
                       
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)        
        optimizer.step()   

        pbar.set_postfix({'Experiment': opt.exp_name,
                'Total loss': total_loss.item(),
                'Mask loss': mask_loss.item(),
                'Token_loss ': token_loss.item(),
                'Embeddings loss ': embeddings_loss.item()})  
        
        epoch_count += 1
        epoch_loss += total_loss  

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss.item() / len(train_loader)))

def validate(opt, epoch, val_loader, sam_transform, model):
    model.eval()
    epoch_loss = 0      
    epoch_count = 0
    
    pbar = tqdm(val_loader)
    for batch in pbar:
        clear_im, degraded_im, clear_fname, all_gt_masks, points, labels = batch
        
        clear_im = clear_im.cuda()
        degraded_im = degraded_im.cuda()
        gt_mask = all_gt_masks.cuda() 
        points = points.cuda()
        labels = labels.cuda()
        
        original_im = torch.permute(clear_im, (0, 3, 1, 2))  
        degraded_im = torch.permute(degraded_im, (0, 3, 1, 2)) 

        all_im = torch.cat((original_im, degraded_im), dim=0) 
        all_im_transformed = sam_transform.apply_image_torch(all_im.float())  
        
        all_points = torch.cat((points, points), dim=0)
        all_labels = torch.cat((labels, labels), dim=0)         

        batched_input = [] 

        for i in range(all_im_transformed.shape[0]):
            data_dict = {}
            data_dict['image'] = all_im_transformed[i]
            data_dict['point_coords'] = sam_transform.apply_coords_torch(all_points[i], all_im.shape[-2:]).unsqueeze(0)
            data_dict['point_labels'] = all_labels[i].unsqueeze(0)
            data_dict['original_size'] = all_im.shape[-2:]       
            
            batched_input.append(data_dict)
            
        with torch.no_grad():               
            batched_output = model(opt, batched_input, multimask_output=False, return_logits=True) 
        
        degraded_index = int(0.5 * len(batched_input))

        # get model output of clear images
        clear_masks = batched_output[0]['masks'] 
        clear_embeddings = batched_output[0]['robust_embeddings']
        clear_tokens = batched_output[0]['robust_token']
        for i in range(1, degraded_index):
            clear_masks = torch.cat((clear_masks, batched_output[i]['masks']), dim=0)
            clear_embeddings = torch.cat((clear_embeddings, batched_output[i]['robust_embeddings']), dim=0)
            clear_tokens = torch.cat((clear_tokens, batched_output[i]['robust_token']), dim=0)  
       
        degraded_masks = batched_output[degraded_index]['masks']
        degraded_embeddings = batched_output[degraded_index]['robust_embeddings']
        degraded_tokens = batched_output[degraded_index]['robust_token']        
        for i in range(degraded_index+1, len(batched_output)):
            degraded_masks = torch.cat((degraded_masks, batched_output[i]['masks']), dim=0)
            degraded_embeddings = torch.cat((degraded_embeddings, batched_output[i]['robust_embeddings']), dim=0)
            degraded_tokens = torch.cat((degraded_tokens, batched_output[i]['robust_token']), dim=0)
        
        gt_mask = gt_mask.float().unsqueeze(1)     
        dice_loss = criterion_dice(degraded_masks, clear_masks.float())          
        focal_loss = criterion_focal(degraded_masks, clear_masks.float())  
        mask_loss_clear = weight_focal*focal_loss + weight_dice*dice_loss
        
        dice_loss = criterion_dice(degraded_masks, gt_mask)          
        focal_loss = criterion_focal(degraded_masks, gt_mask)       
        mask_loss_gt = weight_focal*focal_loss + weight_dice*dice_loss    
           
        mask_loss = mask_loss_gt        
        token_loss = criterion_token(degraded_tokens, clear_tokens)*weight_token
        embeddings_loss = criterion_embeddings(degraded_embeddings, clear_embeddings)*weight_embedding
        
        total_loss = mask_loss + token_loss + embeddings_loss
        pbar.set_postfix({'Experiment': opt.exp_name,
                'Total loss': total_loss.item(),
                'Mask loss': mask_loss.item(),
                'Token_loss ': token_loss.item(),
                'Embeddings loss ': embeddings_loss.item()})  
            
        epoch_count += 1
        epoch_loss += total_loss
   
    return epoch_loss.item() / len(val_loader)