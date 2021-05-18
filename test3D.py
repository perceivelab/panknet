#test  for 3D Model

import os
import torch
from torchsummary import summary
from tqdm import tqdm

import json
from torch.cuda.amp import autocast
from monai.metrics import DiceMetric, ConfusionMatrixMetric, HausdorffDistanceMetric
from monai.data import NiftiSaver
from getModel import get_model
from monai.transforms import (
    AddChannelD,
    AsDiscrete,
    Activations,
    Compose,
    LoadImageD,
    OrientationD,
    SpacingD,
    ToTensorD,
    NormalizeIntensityD,
    ScaleIntensityD
    )
from monai.inferers import sliding_window_inference
import utils.transforms
from dataset.PanDatasetCV import PanDataset as DS

model_list = {'S3DSegPan3D_3D': {'model_name': 'S3DSegPan3D_3D', 'num_slices': 48, 'pretrained' : False },
              'MobilNetSegPan3D_3D' : {'model_name': 'MobilNetSegPan3D_3D', 'num_slices': 48, 'pretrained' : False },
              }

model_name = 'S3DSegPan3D_3D'

model_dict = get_model(model_list[model_name])

image_type = 'MRI'#'CT'

#def main():
image_crop = 256
sw_batch_size = 1

dev = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

include_background = True

metric_names = ('sensitivity', 'specificity', 'accuracy', 'precision', 'f1 score', 'false negative rate', 'false positive rate')

testName = f'test_{include_background}'
train_name = os.path.join(image_type, model_name,'2021-03-02_08-48-31_FOLD3')
test_path = os.path.join('output',train_name, testName)
prediction_path = os.path.join(test_path, 'seg_mask')
weight_name = f'weight_bestDice{include_background}.pt'
file_weight = os.path.join('output', train_name, weight_name)

print('Loading training checkpoint from', train_name)

with open(os.path.join('output', train_name, 'check_point.json')) as fp:
        checkpoint=json.load(fp)
        
model_parameters = model_dict['model_params']

test_parameters = {
    'batch_size' : 1,
    'num_slices': 48,
    'num_fold': checkpoint['num_fold'],
    'best dice epoch {include_background}': checkpoint[f'best_dice_epoch_{include_background}'],
    'shuffle': False, 
    'num_workers': 0,
    }

if image_type == 'CT':
    
    data_dir = os.path.join('data','PANCREAS-CT')
    split_path =  os.path.join(data_dir, 'CT4FoldCrossValSplit.json')
    
    image_key = 'image'
    mask_key = 'mask'
    KEYS = (image_key, mask_key)


    test_transforms = Compose([
        LoadImageD(KEYS),
        AddChannelD(KEYS),
        NormalizeIntensityD(keys = (image_key)),#added for CT
        ScaleIntensityD(keys = (image_key)),#added for CT
        SpacingD(KEYS, pixdim=(1., 1., 1.), mode = ('bilinear', 'nearest')),
        OrientationD(KEYS, axcodes = 'RAS'),
        ToTensorD(KEYS),
    ])

elif image_type == 'MRI':
    import utils
    data_dir = os.path.join('data','PanMRIDataset')
    split_path =  os.path.join(data_dir, 'MRI4FoldCrossValSplit.json')
    
    image_key = 'image_T2'
    mask_key = 'mask'
    KEYS = (image_key, mask_key)

    test_transforms = Compose([
        LoadImageD(KEYS),
        utils.transforms.CopyMetaDictD(KEYS),
        AddChannelD(KEYS),
        SpacingD(KEYS, pixdim=(1., 1., 1.), mode = ('bilinear', 'nearest')),
        OrientationD(KEYS, axcodes = 'RAS'),
        ToTensorD(KEYS),
    ])

else:
    raise 'Error! Image type unrecognized!'




hausdorff_metric = HausdorffDistanceMetric(include_background=include_background, reduction='mean')
dice_metric = DiceMetric(include_background=include_background, reduction="mean")
post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])
discr = AsDiscrete(threshold_values = True, logit_thresh = 0.01)
one_hot = AsDiscrete(to_onehot = True, n_classes=2)
conf_matr_metric = ConfusionMatrixMetric(include_background=include_background, metric_name = metric_names, compute_sample=True)

test_dataset =DS(root_dir = data_dir, split_path = split_path, section = 'test', num_fold = test_parameters['num_fold'], transforms = test_transforms)

model = model_dict['model'].to(dev)

weight_dict = torch.load(file_weight, map_location = dev)
model.load_state_dict(weight_dict, strict = True)

torch.backends.cudnn.benchmark = True
model.eval()

if not os.path.isdir(test_path):
    os.makedirs(test_path)
if not os.path.isdir(prediction_path):
    os.makedirs(prediction_path)

#saving traing info
summ = summary(model, input_data=(1,image_crop, image_crop, test_parameters['num_slices']), device= dev, verbose=0)

info=['model_name: ', model.__class__.__name__ ,'\n',
      'data_dir: ', str(data_dir), '\n',
      'image_crop: ',  str(image_crop),'\n',
      'sw_batch_size: ', str(sw_batch_size), '\n',
      'test_parameters:', str(test_parameters), '\n',
      'model_summary: ','\n', str(summ), '\n'
      ]

file_info=open(os.path.join(test_path, "info.txt"), 'w', encoding='utf-8')
file_info.writelines(info)
file_info.close()

metrics_per_el = {}

with torch.no_grad():
    for el in tqdm(test_dataset):    
        # mri_imgs shape: 1, H, W, D
        # labels shape: 1, H, W, D
        img, labels, img_meta_dict, mask_meta_dict = el[image_key].unsqueeze(0).to(dev), el[mask_key].unsqueeze(0).to(dev), el[f'{image_key}_meta_dict'], el[f'{mask_key}_meta_dict']
        # mri_imgs shape: 1, 1, H, W, D
        # labels shape: 1, 1, H, W, D
        if image_type == 'CT':
            name = img_meta_dict['filename_or_obj'].split(os.sep)[-1].split('.')[0]
        elif image_type == 'MRI':
            name = img_meta_dict['filename_or_obj'].split(os.sep)[-2]
        print(f'{name}')
        roi_size = (image_crop, image_crop, test_parameters['num_slices'])
        sw_batch_size = 4
        
        onehot_labels=one_hot(discr(labels))
        
        with autocast():
            out= sliding_window_inference(img, roi_size, sw_batch_size, model)
        binary_out = post_trans(out)
        onehot_y_pred = one_hot(post_trans(out))
        
        metrics_per_el[name] = {'dice_score': -1., 'hausdorff_distance':-1.}
        
        dice, _ = dice_metric(onehot_y_pred, onehot_labels)
        hausdorff, _ = hausdorff_metric(onehot_y_pred, onehot_labels)
        metrics = conf_matr_metric(onehot_y_pred, onehot_labels)
        
        metrics_per_el[name]['dice_score'] = dice.item()  
        metrics_per_el[name]['hausdorff_distance'] = hausdorff.item()  
    
        for i,m in enumerate(metric_names):       
            metrics_per_el[name][m]= metrics[2*i].item()
            
        if not os.path.isdir(os.path.join(prediction_path, name)):
            os.makedirs(os.path.join(prediction_path, name))   
            
        '''SAVE NIFTI '''
        print('Nifti Saving...')
        nii_saver = NiftiSaver(output_dir = os.path.join(prediction_path,name, 'NIFTI'))
        nii_saver.save(img.squeeze(0), img_meta_dict)
        print('Image saved')
        nii_saver.save(labels.squeeze(0), mask_meta_dict)
        print('Label saved')
        pred_dict = mask_meta_dict.copy()
        pred_dict['filename_or_obj'] = os.path.join(os.path.split(pred_dict['filename_or_obj'])[0], f'{name}_PRED.nii.gz')
        nii_saver.save(binary_out.squeeze(0), pred_dict)
        print('PRED saved')
        print('Saving completed!')
        
with open(os.path.join(test_path, 'metrics.json'), 'w') as fp:
    json.dump(metrics_per_el, fp)
    


























