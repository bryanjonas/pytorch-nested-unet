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
    
    def map_weights(self, mask, w0=10, sigma=5):
        """
        Create a UNet weight map from a boolean `mask` where `True`
        marks the interior pixels of an instance.
        """
        import scipy.ndimage as ndi
        import numpy as np
        mask = mask.reshape(mask.shape[0], mask.shape[1])
    
        #Array must be boolean
        mask = (mask > 0).astype(int)
    
        # if the mask only has one contiguous class,
        # then there isn't much to do.
        if len(np.unique(mask)) == 1:
            mask = mask.reshape(mask.shape[0], mask.shape[1])
            return np.ones(mask.shape, dtype=np.float32) * 0.5

        # calculate the class-balanced weight map w_c
        w_c = np.zeros(mask.shape, dtype=np.float32)
        w_1 = 1 - float(np.count_nonzero(mask)) / w_c.size
        w_0 = 1 - w_1

        # calculate the distance-weighted emphases w_e
        segs, _ = ndi.label(mask)
        if segs.max() == 1:
            # if there is only 1 instance plus background,
            # then there are no separations
            return w_c

        ilabels = range(1, segs.max()+1)
        distmaps = np.stack([ndi.distance_transform_edt(segs != l) for l in ilabels])
        distmaps = np.sort(distmaps, axis=0)[:2]

        w_e = w0 * np.exp((-1 * (distmaps[0] + distmaps[1]) ** 2) / (2 * (sigma ** 2)))
        w_e[mask] = 0.
    
        weight_map = w_e
    
        weight_map = weight_map.reshape(weight_map.shape[0], weight_map.shape[1])
    
        return weight_map
    
    def find_edges(self, mask):
        import scipy.ndimage as ndi

        mask = np.array(mask / 255, dtype=np.bool)
        edge = ndi.binary_dilation(mask)
        edge[mask] = 0
        return edge
    
    def __getitem__(self, idx):
        from skimage.io import imread, imsave
        import numpy as np
        img_id = self.img_ids[idx]
        
        img = imread(os.path.join(self.img_dir, img_id + self.img_ext))
        
        img = img.reshape(img.shape + (1,))
        
        img = img / 2**11
        
        #img_m = img.mean()
        #img_sd = img.std()+1e-12
        
        #img_m = (465.44523522 / 2**11)
        #img_sd = (166.47386568 / 2**11)
        #norm_trans = transforms.Normalize(mean = img_m,
        #                                  std = img_sd,
        #                                  max_pixel_value = (2**11)-1, 
        #                                  always_apply = True)
        
        #augmented = norm_trans(image = img)
        #img = augmented['image']
        
        #img = (img - img_m) / (img_sd)
        
        #Hyperbolic Tangent Normalization    
        #img = 0.5 * (np.tanh((0.01 * (img - img_m))/img_sd) + 1)
        
        #print(img.max())
        #print(img.min())
        
        #for chan in range(0, img.shape[2]):
        #    chan_mean = img[:,:,chan].mean()
        #    chan_sd = img[:,:,chan].std()
        #    img[:,:,chan] = (img[:,:,chan] * chan_mean) / chan_sd
        
        mask_list = []
        #for i in range(self.num_classes):
        mask = cv2.imread(os.path.join(self.mask_dir, "0",
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)
        #mask_list += [mask]
        edge = self.find_edges(mask) * 255
        mask_list += [edge]
        #weight = self.map_weights(mask, w0=1, sigma=6)
        #weight = weight > weight.mean()
        #weight = weight * 255
        #mask_list += [weight]
        mask = np.array(mask_list)
        mask = mask.transpose(1, 2, 0)
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            

        
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        
        return img, mask, {'img_id': img_id}
