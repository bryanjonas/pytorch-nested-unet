import os

import cv2
import numpy as np
import torch
import torch.utils.data


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

    def __getitem__(self, idx):
        from skimage.io import imread, imsave
        import numpy as np
        img_id = self.img_ids[idx]
        
        img = imread(os.path.join(self.img_dir, img_id + self.img_ext))
    
        img = np.array((img[:,:,(4,2,1)]), dtype=np.float32)
        #img = np.array(img*256, dtype=np.uint8)
        mean_list = (331.35564105,463.62619325,357.02413693)
        std_list = (124.93471721,133.2585851,84.44538196)
        
        for chan in range(0,3):
            img[:,:,chan] = (img[:,:,chan] - mean_list[chan]) / std_list[chan]
        
        mask = []
        for i in range(self.num_classes):
            img_mask = cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)
            #img_mask = 255 - img_mask 
            mask.append(img_mask[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        #print(img.max())
        #print(img.min())
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        
        return img, mask, {'img_id': img_id}
