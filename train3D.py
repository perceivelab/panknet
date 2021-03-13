import os
import sys
import torch
from torch.cuda.amp import autocast, GradScaler

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from datetime import datetime
import time
import json
from tqdm import tqdm
from getModel import get_model

import monai
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, list_data_collate
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import (
    AddChannelD,
    AsDiscrete,
    Activations,
    Compose,
    LoadImageD,
    OrientationD,
    RandFlipD,
    RandRotate90D,
    RandCropByPosNegLabelD,
    SpacingD,
    ToTensorD, 
    NormalizeIntensityD, 
    ScaleIntensityD
    )

from dataset.PanDatasetCV import PanDataset as DS

model_list = {'S3DSegPan3D_3D': {'model_name': 'S3DSegPan3D_3D', 'num_slices': 48, 'pretrained' : True },
             'MobilNetSegPan3D_3D' : {'model_name': 'MobilNetSegPan3D_3D', 'num_slices': 48, 'pretrained' : True },
              }

model_name = 'MobilNetSegPan3D_3D'

model_dict = get_model(model_list[model_name])

image_type = 'MRI'#'CT'

'''The Input of the model should be NCHWD (N1HWD)
)'''

num_fold = 3

image_crop = 128

output_folder = 'output'
subfolder = os.path.join(image_type,model_name)
timestamp_str = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
train_name = timestamp_str+f'_Concetto_FOLD{num_fold}_noPretrainedCT'

include_background_loss = True

fine_tuning  = False
continue_training = False
original_train_name = '' if continue_training or fine_tuning else ''

if fine_tuning:
    train_name = train_name + f'_fineTuning{image_type}'

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_parameters = model_dict['model_params']

train_parameters = {
    'batch_size': 8, 
    'num_slices': 48,
    'center': -1,
    'num_fold': num_fold,
    'image_crop': (image_crop, image_crop),
    'shuffle': True, 
    'num_workers': 0, 
    'num_epochs' : 3000, 
    'learning_rate' : 1e-4, 
    'fine_Tuning': fine_tuning,
    'continue_training': continue_training,
    'include_background_loss': include_background_loss,
    'original_train_name': original_train_name, 
    'pos_RandCrop' : 0.8,
    'neg_RandCrop': 0.2,
    'transformation' : "LoadImageD(KEYS), AddChannelD(KEYS), NormalizeIntensityD(keys = (image_key)), ScaleIntensityD(keys = (image_key)),SpacingD(KEYS, pixdim=(1., 1., 1.), mode = ('bilinear', 'nearest')), OrientationD(KEYS, axcodes = 'RAI'), RandCropByPosNegLabelD(keys = KEYS, label_key=KEYS[1], spatial_size = (image_crop,image_crop,train_parameters['num_slices']), pos = train_parameters['pos_RandCrop'], neg = train_parameters['neg_RandCrop']), RandFlipD(KEYS, prob =0.5, spatial_axis=0), RandRotate90D(KEYS, prob=0.5, spatial_axes=(0,1)),ToTensorD(KEYS)"
    }

val_parameters = {
    'batch_size': 1, 
    'num_fold':train_parameters['num_fold'],
    'val_set': [],
    'test_set': [],
    'shuffle': False, 
    'num_workers': 0, 
    'val_interval': 5,
    'num_slices' : train_parameters['num_slices'],
    'transformations': "LoadImageD(KEYS), AddChannelD(KEYS), NormalizeIntensityD(keys = (image_key)), ScaleIntensityD(keys = (image_key)),SpacingD(KEYS, pixdim=(1., 1., 1.), mode = ('bilinear', 'nearest')),OrientationD(KEYS, axcodes = 'RAI'), ToTensorD(KEYS),"
    }

if image_type == 'CT':
    
    data_dir = os.path.join('data','PANCREAS-CT')
    split_path =  os.path.join(data_dir, 'CT4FoldCrossValSplit.json')
    image_key = 'image'
    mask_key = 'mask'
    KEYS = (image_key, mask_key) 

    train_transforms = Compose([
        LoadImageD(KEYS),
        AddChannelD(KEYS),
        NormalizeIntensityD(keys = (image_key)),
        ScaleIntensityD(keys = (image_key)),
        SpacingD(KEYS, pixdim=(1., 1., 1.), mode = ("bilinear", "nearest")),
        OrientationD(KEYS, axcodes = 'RAS'),
        RandCropByPosNegLabelD(keys = KEYS, label_key=KEYS[1], spatial_size = (image_crop,image_crop,train_parameters['num_slices']), pos = train_parameters['pos_RandCrop'], neg = train_parameters['neg_RandCrop']),
        RandFlipD(KEYS, prob = 0.5, spatial_axis=0),
        RandRotate90D(KEYS, prob=0.5, spatial_axes=(0,1)),    
        ToTensorD(KEYS)
        ])
    
    val_transforms = Compose([
        LoadImageD(KEYS),
        AddChannelD(KEYS),
        NormalizeIntensityD(keys = (image_key)),
        ScaleIntensityD(keys = (image_key)),
        SpacingD(KEYS, pixdim=(1., 1., 1.), mode = ('bilinear', 'nearest')),
        OrientationD(KEYS, axcodes = 'RAS'),
        ToTensorD(KEYS),
        ])

elif image_type == 'MRI':
    import utils.transforms
    data_dir = os.path.join('data','PanMRIDataset')
    split_path =  os.path.join(data_dir, 'MRI4FoldCrossValSplit.json')
    
    image_key = 'image_T2'
    mask_key = 'mask'
    KEYS = (image_key, mask_key)

    train_transforms = Compose([
        LoadImageD(KEYS),
        utils.transforms.CopyMetaDictD(KEYS),
        AddChannelD(KEYS),
        SpacingD(KEYS, pixdim=(1., 1., 1.), mode = ("bilinear", "nearest")),
        OrientationD(KEYS, axcodes = 'RAS'),
        RandCropByPosNegLabelD(keys = KEYS, label_key=KEYS[1], spatial_size = (image_crop,image_crop,train_parameters['num_slices']), pos = train_parameters['pos_RandCrop'], neg = train_parameters['neg_RandCrop']),
        RandFlipD(KEYS, prob = 0.5, spatial_axis=0),
        RandRotate90D(KEYS, prob=0.5, spatial_axes=(0,1)),    
        ToTensorD(KEYS)
        ])
    
    val_transforms = Compose([
        LoadImageD(KEYS),
        utils.transforms.CopyMetaDictD(KEYS),
        AddChannelD(KEYS),
        SpacingD(KEYS, pixdim=(1., 1., 1.), mode = ('bilinear', 'nearest')),
        OrientationD(KEYS, axcodes = 'RAS'),
        ToTensorD(KEYS),
        ])
else:
    raise 'Error! Image type unrecognized!'

check_point={'epoch' : 0,
            'min_loss_val' : sys.float_info.max,
            'min_loss_epoch' : 0,
            'best_dice_score_True' : sys.float_info.min,
            'best_dice_epoch_True' : 0,
            'best_dice_score_False' : sys.float_info.min,
            'best_dice_epoch_False' : 0,
            'num_fold' : train_parameters['num_fold'],
            'loss_history' : {'map1':[], 'map2':[], 'map3':[], 'map4':[], 'final':[], 'total':[] , 'validation':[]},
            'dice_score_val_history_False':[],
            'dice_score_val_history_True':[],
            'dice_score_test_history_False':[],
            'dice_score_test_history_True':[]
        }

if continue_training:
    print('Loading checkpoint to continue training from ', original_train_name)
    with open(os.path.join(output_folder, subfolder, original_train_name, 'check_point.json')) as fp:
            check_point=json.load(fp)
    train_parameters['num_fold'] = check_point['num_fold']
    val_parameters['num_fold'] = check_point['num_fold']

dataset ={
    'train': DS(root_dir = data_dir, split_path = split_path, section = 'training', num_fold = train_parameters['num_fold'], transforms = train_transforms),
    'val': DS(root_dir = data_dir, split_path = split_path, section = 'validation', num_fold = val_parameters['num_fold'], transforms = val_transforms),
    }

#storing validation set
for i in range(len(dataset['val'])):
    val_parameters['val_set'].append(dataset['val'][i][f'{KEYS[0]}_meta_dict']['filename_or_obj'].split(os.sep)[-1] )
'''
#storing test set
for i in range(len(dataset['test'])):
    val_parameters['test_set'].append(dataset['test'][i][f'{KEYS[0]}_meta_dict']['filename_or_obj'].split(os.sep)[-1])
'''

#check dataset and DataLoader
check_loader = DataLoader(dataset['val'], batch_size=1, shuffle=False, num_workers=0)
check_data = monai.utils.misc.first(check_loader)
print(check_data[KEYS[0]].shape, check_data[KEYS[1]].shape)

train_loader = DataLoader(dataset['train'], batch_size=train_parameters['batch_size'], shuffle=train_parameters['shuffle'], num_workers=train_parameters['num_workers'], collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(dataset['val'], batch_size=val_parameters['batch_size'], shuffle=val_parameters['shuffle'], num_workers=val_parameters['num_workers'], collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

model = model_dict['model'].to(dev)

diceMetricTrue = DiceMetric(include_background=True, reduction="mean")
diceMetricFalse = DiceMetric(include_background=False, reduction="mean")

post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

#post-transformation of labels
discretize_labels = AsDiscrete(threshold_values = True, logit_thresh = 0.01)
one_hot = AsDiscrete(to_onehot = True, n_classes=2)

loss_function = monai.losses.DiceLoss(softmax=True, include_background=train_parameters['include_background_loss'])

optimizer = torch.optim.Adam(model.parameters(), lr = train_parameters['learning_rate'])
scaler = GradScaler()

path_tensorboard= os.path.join('runs', subfolder, train_name)
writer = SummaryWriter(path_tensorboard)

#saving traing info
summ = summary(model, 
               input_data=(1,image_crop, image_crop, train_parameters['num_slices']), 
               device= dev, 
               verbose=0, 
               depth = 10, 
               col_names = ['input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'])

info=['model_name: ', model.__class__.__name__ ,'\n',
      'data_dir: ', str(data_dir), '\n',
      'image_crop: ',  str(image_crop),'\n',
      'train_parameters:', str(train_parameters), '\n',
      'val_parameters:', str(val_parameters), '\n',
      'model_parameters:', str(model_parameters), '\n',
      'model_summary: ','\n', str(summ), '\n']

if not os.path.isdir(os.path.join('output', subfolder, train_name)):
        os.makedirs(os.path.join('output', subfolder, train_name))
        
file_info=open(os.path.join("output", subfolder, train_name, "train_info.txt"), 'w', encoding='utf-8')
file_info.writelines(info)
file_info.close()
    

#continue_training
if continue_training:
    print('Loading weights to continue training from ', original_train_name)
            
    # load file weight 
    weight_name = 'weight.pt'
    file_weight = os.path.join('output',subfolder, original_train_name, weight_name)
    
    optim_name='optim.pt'
    file_optimizer=os.path.join('output',subfolder, original_train_name, optim_name)
    
    # load file weight adn optimizer
    model.load_state_dict(torch.load(file_weight, map_location=dev)) #caricamento pesi
    optimizer.load_state_dict(torch.load(file_optimizer))
    
    
    print('Upload tensorboard value...')
    for key in tqdm(check_point['loss_history']):
        if not 'validation' in key:
            tag = 'train/'+key+'_epoch_loss'
            for ep, val in enumerate(check_point['loss_history'][key]):
                writer.add_scalar(tag , val, ep)
        else:
            tag = 'val_epoch_loss'
            for ep, val in enumerate(check_point['loss_history'][key]):
                writer.add_scalar(tag , val, ep*val_parameters['val_interval'])
        
    for i in tqdm(range(len(check_point['dice_score_val_history_False']))):
        writer.add_scalar('dice_score/val' , 
                           check_point['dice_score_val_history_True'][i],
                           i*val_parameters['val_interval'])
        writer.add_scalar('dice_score/valFalse' , 
                           check_point['dice_score_val_history_False'][i], 
                           i*val_parameters['val_interval'])

    print('Uploaded!')
    print('Loaded successfully!')
elif fine_tuning:
    print('Loading weights to finetune from ', original_train_name)
            
    # load file weight (fine-tuning)
    weight_name = 'weight_bestDiceTrue.pt'
    file_weight = os.path.join('output', 'CT', model_name, original_train_name, weight_name)
    
    # load file weight adn optimizer
    model.load_state_dict(torch.load(file_weight, map_location=dev)) #loading weights



epoch_len = len(dataset['train']) // train_loader.batch_size
for epoch in range(check_point['epoch'],train_parameters['num_epochs']):
    print("-"*10)
    print(f"{subfolder}-{train_name}, epoch {epoch+1}/{train_parameters['num_epochs']}")
    model.train()
    epoch_loss = {'map1':0, 'map2':0, 'map3':0, 'map4':0, 'final':0, 'total':0}
    step = 0
    for batch_data in train_loader:
        total_loss = 0
        step +=1
        # inputs shape: B, 1, H, W, D
        # masks shape: B, 1, H, W, D
        inputs, labels = batch_data[image_key].to(dev), batch_data[mask_key].to(dev)
        
        labels = one_hot(discretize_labels(labels))
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs) # outputs is a dict of maps, singol map shape: B, 2, H, W, D
            for key, out in outputs.items():
                loss = loss_function(out.to(dev), labels.to(dev))
                epoch_loss[key] = epoch_loss[key] + loss.item()
                total_loss = total_loss + loss
            epoch_loss['total'] = epoch_loss['total']+total_loss.item()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"{step}/{epoch_len}, train_loss: {total_loss.item():.4f}")
        writer.add_scalar("train/total_iter_loss", total_loss.item(), epoch_len * epoch + step)
    check_point['epoch'] += 1
    for key, out in epoch_loss.items():
        mean_epoch_loss = epoch_loss[key]/step
        check_point['loss_history'][key].append(mean_epoch_loss)
        writer.add_scalar(f"train/{key}_epoch_loss", mean_epoch_loss, epoch)
    print(f"epoch {epoch + 1} average loss: {check_point['loss_history']['total'][-1]:.4f}")
    
    
    #validation
    if (epoch+1)%val_parameters['val_interval'] ==0:
        print('Validation...')
        model.eval()
        with torch.no_grad():
            loss_val = 0
            loss_val_sum = 0
            val_step = 0
            metric_sumT = 0.0
            metric_sumF = 0.0
            metric_countT  = 0
            metric_countF  = 0
            for batch_data in tqdm(val_loader):
                val_step+=1
                # inputs shape: 1, 1, H, W, D
                # masks shape: 1, 1, H, W, D
                val_inputs, val_labels = batch_data[image_key].to(dev), batch_data[mask_key].to(dev)
                roi_size = (image_crop, image_crop, val_parameters['num_slices'])
                sw_batch_size = 1
            
                onehot_val_labels=one_hot(discretize_labels(val_labels))
                with autocast():
                    out_val= sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                
                # compute overall mean dice
                onehot_y_pred = one_hot(post_trans(out_val))
                valueF,notnF = diceMetricFalse(y_pred=onehot_y_pred, y=onehot_val_labels)
                valueT,notnT = diceMetricTrue(y_pred=onehot_y_pred, y=onehot_val_labels)
                metric_countT += len(valueT)
                metric_countF += len(valueF)
                metric_sumT += valueT.item()*len(valueT)
                metric_sumF += valueF.item()*len(valueF)
                #compute loss
                loss_val = loss_function(out_val, onehot_val_labels)
                loss_val_sum += loss_val.item()
            
            metricT = metric_sumT / metric_countT 
            metricF = metric_sumF / metric_countF 
            check_point['dice_score_val_history_True'].append(metricT)
            check_point['dice_score_val_history_False'].append(metricF)
            writer.add_scalar("dice_score/val",metricT,epoch)
            writer.add_scalar("dice_score/valFalse",metricF,epoch)  
            
            if metricT > check_point['best_dice_score_True']:
                check_point['best_dice_score_True'] = metricT
                check_point['best_dice_epoch_True'] = epoch+1
                torch.save(model.state_dict(), os.path.join('output', subfolder,train_name,"weight_bestDiceTrue.pt"))
                torch.save(optimizer.state_dict(), os.path.join('output', subfolder, train_name, 'optim_bestDiceTrue.pt'))
                print("saved new best dice True score model")
                
            if metricF > check_point['best_dice_score_False']:
                check_point['best_dice_score_False'] = metricF
                check_point['best_dice_epoch_False'] = epoch+1
                torch.save(model.state_dict(), os.path.join('output', subfolder,train_name,"weight_bestDiceFalse.pt"))
                torch.save(optimizer.state_dict(), os.path.join('output', subfolder, train_name, 'optim_bestDiceFalse.pt'))
                print("saved new best dice False score model")
            
            epoch_val_loss = loss_val_sum/val_step
            check_point['loss_history']['validation'].append(epoch_val_loss)
            writer.add_scalar("val_epoch_loss", epoch_val_loss, epoch) 
           
            if epoch_val_loss < check_point['min_loss_val']:
                check_point['min_loss_val'] = epoch_val_loss
                check_point['min_loss_epoch'] = epoch+1
                torch.save(model.state_dict(), os.path.join('output', subfolder,train_name,"weight_minloss.pt"))
                torch.save(optimizer.state_dict(), os.path.join('output', subfolder, train_name, 'optim_minloss.pt'))
                print("saved new min val_loss model")
            print("epoch: {} current val_loss: {:.4f} best val_loss: {:.4f} at epoch {} best dice_score_True {:.4f} at epoch {}, best dice_score_False {:.4f} at epoch {}".format(
                        epoch + 1, epoch_val_loss, check_point['min_loss_val'], check_point['min_loss_epoch'], check_point['best_dice_score_True'], check_point['best_dice_epoch_True'],check_point['best_dice_score_False'], check_point['best_dice_epoch_False'])
                )
            print('Validation Completed!')
            
            '''
                SAVE WEIGHTs AND CHECKPOINT
            '''
            torch.save(model.state_dict(), os.path.join('output', subfolder, train_name, 'weight.pt'))
            torch.save(optimizer.state_dict(), os.path.join('output', subfolder, train_name, 'optim.pt'))
            
            with open(os.path.join('output', subfolder, train_name, 'check_point.json'), 'w') as fp:
                json.dump(check_point, fp)
            
            if((epoch+1)%20==0):
                compose_image = torch.cat((val_inputs, val_labels, post_trans(out_val)),dim = 3)
                plot_2d_or_3d_image(compose_image, epoch + 1, writer, index=0, tag="Plot/image", max_frames=200)
            
print(f"train completed, best_metric: {check_point['min_loss_val']:.4f} at epoch: {check_point['min_loss_epoch']}")
        
        
        