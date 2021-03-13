import numpy as np
import copy
from monai.transforms import MapTransform, Randomizable
from monai.config import KeysCollection

class TransposeITKD(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            data[key]=data[key].transpose((1,2,0))
        return data


class CopyMetaDictD(MapTransform):
    def __call__(self, data):
        i,m = self.keys
        m_name = data[f'{m}_meta_dict']['filename_or_obj']
        data[f'{m}_meta_dict'] = copy.deepcopy(data[f'{i}_meta_dict']) 
        data[f'{m}_meta_dict']['filename_or_obj'] = m_name
        data[f'{m}'] = data[f'{m}'].squeeze(3)
        return data

class CopyAffineFromImageITKD(MapTransform):
    def __call__(self, data):
        i,m = self.keys
        data[f'{m}_meta_dict']['affine'] = data[f'{i}_meta_dict']['affine']
        data[f'{m}_meta_dict']['original_affine'] = data[f'{i}_meta_dict']['original_affine']
        data[f'{m}_meta_dict']['spacing'] = data[f'{i}_meta_dict']['spacing']
        for j in range(8):
            data[f'{m}_meta_dict'][f'pixdim[{j}]'] = data[f'{i}_meta_dict'][f'pixdim[{j}]']
        return data
    
class RandDepthCrop(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, num_slices=8, negative_frac=0.0):
        super().__init__(keys)
        self.num_slices = num_slices
        self.negative_frac = negative_frac
        self.start_id = 0
   
    def randomize(self, data):
        self.start_id = int(self.R.random_sample() * data)
    
    def __call__(self, data):
        i, m = self.keys
        mask_max = -0.1
        d_max = data[i].shape[3] - self.num_slices
        rand = self.R.random_sample()
        
        if rand < self.negative_frac:
            self.randomize(d_max)
            slice_ = data[i][0,:,:,self.start_id:(self.start_id+self.num_slices)]
            mask = data[m][0,:,:,(self.start_id+self.num_slices)]
        else:
            while mask_max<= 0.0:
                self.randomize(d_max)
                slice_ = data[i][:,:,:,self.start_id:(self.start_id+self.num_slices)]
                mask = data[m][:,:,:,(self.start_id+self.num_slices)]
                mask_max = mask.max()
        
        cropper = copy.deepcopy(data) 
        cropper[i] = slice_
        cropper[m] = np.expand_dims(mask, axis=3)
        cropper['target_id'] = self.start_id+self.num_slices
        
        #print(cropper[img].shape)
        #print(cropper[mask].shape)
        return cropper


class RandDepthCropCentered(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, num_slices=8, center=3, negative_frac=0.0):
        super().__init__(keys)
        assert (num_slices>=center), 'Error num_slices must be grater then center'
        self.num_slices = num_slices
        self.negative_frac = negative_frac
        self.start_id = 0
        self.center = center
   
    def randomize(self, data):
        self.start_id = int(self.R.random_sample() * data)
    
    def __call__(self, data):
        i, m = self.keys
        mask_max = -0.1
        d_max = data[i].shape[3] - self.num_slices
        rand = self.R.random_sample()
        
        if rand < self.negative_frac:
            self.randomize(d_max)
            slice_ = data[i][:,:,:,self.start_id:(self.start_id+self.num_slices)]
            if self.center == -1:
                mask = data[m][:,:,:,self.start_id:(self.start_id+self.num_slices)]
                target_id = -1
            else:
                mask = data[m][:,:,:,(self.start_id+self.center)]
                mask = np.expand_dims(mask, axis=3)
                target_id = self.start_id+self.center
        else:
            while mask_max<= 0.0:
                self.randomize(d_max)
                slice_ = data[i][:,:,:,self.start_id:(self.start_id+self.num_slices)]
                if self.center == -1:
                    mask = data[m][:,:,:,self.start_id:(self.start_id+self.num_slices)]
                    target_id = -1
                else:
                    mask = data[m][:,:,:,(self.start_id+self.center)]
                    mask = np.expand_dims(mask, axis=3)
                    target_id = self.start_id+self.center
                mask_max = mask.max()
        
        cropper = copy.deepcopy(data) 
        cropper[i] = slice_
        cropper[m] = mask
        
        cropper['start_id'] = self.start_id
        cropper['target_id'] = target_id
        return cropper