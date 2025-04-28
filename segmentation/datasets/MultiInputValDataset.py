from builtins import FileNotFoundError
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.ade import ADE20KDataset
import os
import numpy as np
import torch
import mmcv


@DATASETS.register_module()
class MultiInputTypePerClassADE20KDataset(ADE20KDataset):
    def __init__(self, img_dir, ann_dir, mask_root, types=['box', 'scribble', 'dot'], suffix='.npy', pipeline=None, **kwargs):
        super().__init__(img_dir=img_dir, ann_dir=ann_dir, pipeline=pipeline, **kwargs)
        self.mask_root = mask_root
        self.types = types
        self.suffix = suffix
        self.data_items = []
        self.filename_to_index = {}  # Create a dictionary to map filenames to original indices
        for i, img_info in enumerate(self.img_infos):
            self.filename_to_index[img_info['filename']] = i
        self._prepare_data_items()

    def _prepare_data_items(self):
        for img_info in self.img_infos:
            img_name = os.path.splitext(os.path.basename(img_info['filename']))[0]
            split = 'validation' if 'val' in img_info['filename'].lower() else 'training'
            for input_type in self.types:
                mask_dir = os.path.join(f"{self.mask_root}_{input_type}", split)
                mask_files = [f for f in os.listdir(mask_dir) if f.startswith(f"{img_name}_cls") and f.endswith(self.suffix)]
                for mask_file in mask_files:
                    class_id_str = mask_file.split('_cls')[-1].split(self.suffix)[0]
                    try:
                        class_id = int(class_id_str)
                        self.data_items.append((img_info['filename'], input_type, class_id))
                    except ValueError:
                        print(f"Warning: Could not parse class ID from mask file: {mask_file}")
    def __len__(self):
        return len(self.data_items)
    def __getitem__(self, idx):
        img_filename, input_type, class_id = self.data_items[idx]
        original_img_index = self.filename_to_index.get(img_filename)
        
        if original_img_index is not None:
            results = self.prepare_train_img(original_img_index) # Or prepare_test_img?

            # Load the specific mask
            base_name = os.path.splitext(os.path.basename(img_filename))[0]
            split = 'validation' if 'val' in img_filename.lower() else 'training'
            type_dir = os.path.join(f"{self.mask_root}_{input_type}", split)
            mask_file = f"{base_name}_cls{class_id}{self.suffix}"
            mask_path = os.path.join(type_dir, mask_file)

            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                # Ensure it's (H, W) and then add the channel dimension
                if mask.ndim == 2:
                    gt_bbox_masks = torch.from_numpy(mask).float().unsqueeze(0) # (1, H, W)
                elif mask.ndim == 3 and mask.shape[0] == 1:
                    gt_bbox_masks = torch.from_numpy(mask).float() # (1, H, W) - already has channel
                else:
                    print(f"Warning: Unexpected mask shape {mask.shape} for {mask_path}. Assuming single channel.")
                    gt_bbox_masks = torch.from_numpy(mask).float().unsqueeze(0)

                results['gt_bbox_masks'] = gt_bbox_masks
                results['input_type'] = input_type
            else:
                # Handle missing mask
                results['gt_bbox_masks'] = torch.zeros((1, results['img_shape'][0], results['img_shape'][1]), dtype=torch.float32)
                results['input_type'] = input_type

            # Load ground truth semantic segmentation
            if self.ann_dir is not None:
                gt_seg_map = mmcv.imread(
                    self._join_path(self.img_dir, self.seg_prefix,
                                     self.img_infos[original_img_index]['ann']['seg_map']),
                    flag='unchanged',
                    backend=self.img_backend)
                results['gt_semantic_seg'] = gt_seg_map.squeeze().astype(np.uint8)
                if len(np.unique(results['gt_semantic_seg'])) > self.num_classes:
                    raise ValueError(f"Segmentation map contains more than {self.num_classes} classes.")
            print(results.keys())
            return self.pipeline(results)
        else:
            raise FileNotFoundError(f"Image filename '{img_filename}' not found in self.filename_to_index.")