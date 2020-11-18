import os

import cv2
import numpy as np
import torch
import torch.utils.data
from albumentations.augmentations import transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)
    
    def find_edges(self, mask):
        import scipy.ndimage as ndi

        edge = ndi.binary_dilation(mask)
        edge[mask] = 0
        return edge
    
    def __getitem__(self, idx):
        from skimage.io import imread, imsave
        import numpy as np
        img_id = self.img_ids[idx]
        
        img = imread(os.path.join(self.img_dir, img_id + self.img_ext))
        
        #img = img.reshape(img.shape + (1,))
        
        img = (img / 2**11) * 255
        
        #img_m = img.mean()
        #img_sd = img.std()+1e-12
        
        #img_m = (465.44523522 / 2**11)
        #img_sd = (166.47386568 / 2**11)
        
        #img = (img - img_m) / (img_sd)
        
        #print(img.max())
        #print(img.min())
        
        #for chan in range(0, img.shape[2]):
        #    chan_mean = img[:,:,chan].mean()
        #    chan_sd = img[:,:,chan].std()
        #    img[:,:,chan] = (img[:,:,chan] * chan_mean) / chan_sd
        
        bldg_mask = cv2.imread(os.path.join(self.mask_dir, "0",
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)
        
        bldg_mask = np.array(bldg_mask / 255, dtype = np.bool)
        
        edges = self.find_edges(bldg_mask)
        edges.astype(np.bool)
    

        mask = np.ones(shape=bldg_mask.shape)
        mask[bldg_mask] = 0
        mask[edges] = 2
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)
        mask = mask / 255
        mask = mask.astype('int64')
        #mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}
