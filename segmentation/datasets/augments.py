import torch
import random
import numpy as np
from mmseg.datasets.pipelines import Resize, RandomCrop, RandomFlip, Pad, LoadImageFromFile
from mmseg.datasets.builder import PIPELINES

from builtins import FileNotFoundError
import os
import random
import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES




@PIPELINES.register_module()
class LoadPerClassMasksFromFolder:
    def __init__(self, mask_root, types=['box', 'scribble', 'dot'], suffix='.npy', random_select=True):
        self.mask_root = mask_root
        self.types = types
        self.suffix = suffix
        self.random_select = random_select  # new argument!
    
    def __call__(self, results):
        img_filename = os.path.basename(results['img_info']['filename'])
        base_name = os.path.splitext(img_filename)[0]

        full_path = results['img_info']['filename']
        split = 'training' if 'train' in full_path.lower() else 'validation'

        gt_bbox_masks = {}

        for input_type in self.types:
            type_dir = os.path.join(f"{self.mask_root}_{input_type}", split)
            masks = [
                torch.from_numpy(np.load(os.path.join(type_dir, file)))
                for file in sorted(os.listdir(type_dir))
                if file.startswith(base_name + '_cls') and file.endswith(self.suffix)
            ]

            if masks:
                gt_bbox_masks[input_type] = torch.stack(masks, dim=0).float()  # (n_cls, H, W)

        if self.random_select:
            # During training → random one
            
            chosen_type = random.choice(list(gt_bbox_masks.keys()))
            chosen_class = random.randint(0,len(gt_bbox_masks[chosen_type])-1)
            results['gt_bbox_masks'] = gt_bbox_masks[chosen_type][chosen_class].unsqueeze(0)
            results['input_type'] = chosen_type
        else:
            # During validation → load all
            results['gt_bbox_masks'] = gt_bbox_masks
            results['input_type'] = None  # or just set None

        return results


@PIPELINES.register_module()
class ResizeWithBBox(Resize):
    def __init__(self, img_scale=None, multiscale_mode='range', ratio_range=None, keep_ratio=True, min_size=None):
        super().__init__(img_scale, multiscale_mode, ratio_range, keep_ratio, min_size)
    
    def __call__(self, results):
        results = super().__call__(results)
        
        if 'gt_bbox_masks' in results:
            img_shape = results['img_shape']
            masks = results['gt_bbox_masks']
            if isinstance(masks, torch.Tensor):
                results['gt_bbox_masks'] = torch.nn.functional.interpolate(
                    masks.unsqueeze(0).float(), size=img_shape[:2], mode='nearest'
                ).squeeze(0)
        return results

@PIPELINES.register_module()
class RandomCropWithBBox(RandomCrop):
    def __init__(self, crop_size, cat_max_ratio=1, ignore_index=255):
        super().__init__(crop_size, cat_max_ratio, ignore_index)

    def __call__(self, results):
        if 'gt_bbox_masks' not in results:
            return super().__call__(results)

        img = results['img']
        img_h, img_w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        # If image is smaller than crop, skip cropping
        if img_h <= crop_h and img_w <= crop_w:
            return results

        # Randomly sample crop
        max_y = max(0,img_h - crop_h)
        max_x = max(0,img_w - crop_w)
        y1 = random.randint(0, max_y)
        x1 = random.randint(0, max_x)
        y2 = y1 + crop_h
        x2 = x1 + crop_w

        # Crop image
        results['img'] = img[y1:y2, x1:x2, ...]
        results['img_shape'] = results['img'].shape

        # Crop semantic segmentation
        if 'gt_semantic_seg' in results:
            seg = results['gt_semantic_seg']
            results['gt_semantic_seg'] = seg[y1:y2, x1:x2]

        # Crop bbox masks
        masks = results['gt_bbox_masks']
        results['gt_bbox_masks'] = masks[:, y1:y2, x1:x2]

        return results
    
@PIPELINES.register_module()
class RandomFlipWithBBox(RandomFlip):
    def __init__(self, prob=0.0, direction='horizontal'):
        super().__init__(prob, direction)

    def __call__(self, results):
        results = super().__call__(results)
        if 'gt_bbox_masks' in results and results.get('flip', False):
            flip_direction = results.get('flip_direction', 'horizontal')
            masks = results['gt_bbox_masks']
            if flip_direction == 'horizontal':
                results['gt_bbox_masks'] = masks.flip(dims=[2])  # Flip width
            elif flip_direction == 'vertical':
                results['gt_bbox_masks'] = masks.flip(dims=[1])  # Flip height
        return results

@PIPELINES.register_module()
class PadToSizeWithBBox(Pad):
    def __call__(self, results):
        results = super().__call__(results)

        img = results['img']
        img_h, img_w = img.shape[:2] 

        # --- Bounding box masks padding ---
        if 'gt_bbox_masks' in results:
            masks = results['gt_bbox_masks']
            if isinstance(masks, torch.Tensor):
                masks_h, masks_w = masks.shape[-2:]  # (n_cls, H, W)

                pad_h = img_h - masks_h
                pad_w = img_w - masks_w

                if pad_h > 0 or pad_w > 0:
                    masks = torch.nn.functional.pad(
                        masks,
                        (0, pad_w, 0, pad_h),  # pad width then height
                        mode='constant',
                        value=0
                    )
                
                results['gt_bbox_masks'] = masks
            else:
                print("Warning: gt_bbox_masks exists but is not a tensor.")



        return results
    

@PIPELINES.register_module()
class FilterSegMaskToSingleClass:
    def __init__(self, keep_class=42, ignore_index=255):
        self.keep_class = keep_class
        self.ignore_index = ignore_index

    def __call__(self, results):
        seg = results['gt_semantic_seg']
        mask = seg == self.keep_class
        seg[:] = self.ignore_index  # Set everything to ignore
        seg[mask] = self.keep_class  # Set only class 42
        results['gt_semantic_seg'] = seg
        return results