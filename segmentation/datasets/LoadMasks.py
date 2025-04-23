
import os
import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES  # Register custom pipeline

@PIPELINES.register_module()
class LoadPerClassMasksFromFolder:
    def __init__(self, mask_root, types=['box', 'scribble', 'dot'], suffix='.npy'):
        self.mask_root = mask_root
        self.types = types
        self.suffix = suffix

    def __call__(self, results):
        img_filename = os.path.basename(results['img_info']['filename'])
        base_name = os.path.splitext(img_filename)[0]  # e.g., ADE_000001

        gt_bbox_masks = {}

        for input_type in self.types:
            type_dir = os.path.join(self.mask_root, input_type)
            class_masks = []

            # Look for files matching the pattern: ADE_000001_cls*
            for file in sorted(os.listdir(type_dir)):
                if file.startswith(base_name + '_cls') and file.endswith(self.suffix):
                    mask = np.load(os.path.join(type_dir, file))  # shape: (H, W)
                    class_masks.append(torch.from_numpy(mask))

            if not class_masks:
                raise FileNotFoundError(f"No masks found for {base_name} in {input_type}")

            stacked = torch.stack(class_masks, dim=0)  # shape: (n_classes, H, W)
            gt_bbox_masks[input_type] = stacked

        results['gt_bbox_masks'] = gt_bbox_masks
        return results
