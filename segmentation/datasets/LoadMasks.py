
import os
import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES  # Register custom pipeline
import random
@PIPELINES.register_module()
class LoadPerClassMasksFromFolder:
    def __init__(self, mask_root, types=['box', 'scribble', 'dot'], suffix='.npy'):
        self.mask_root = mask_root  # base prefix like /path/to/prompt_masks_
        self.types = types
        self.suffix = suffix

    def __call__(self, results):
        import random
        import numpy as np
        import torch
        import os

        img_filename = os.path.basename(results['img_info']['filename'])  # ADE_train_00000001.jpg
        base_name = os.path.splitext(img_filename)[0]                     # ADE_train_00000001

        # Detect split from path
        full_path = results['img_info']['filename']
        split = 'training' if 'train' in full_path else 'validation'

        gt_bbox_masks = {}

        for input_type in self.types:
            type_dir = os.path.join(f"{self.mask_root}_{input_type}", split)
            class_masks = []

            for file in sorted(os.listdir(type_dir)):
                if file.startswith(base_name + '_cls') and file.endswith(self.suffix):

                    mask = np.load(os.path.join(type_dir, file))
                    class_masks.append(torch.from_numpy(mask))

            if not class_masks:
                raise FileNotFoundError(f"No masks found for {base_name} in {input_type}")

            stacked = torch.stack(class_masks, dim=0)  # (n_cls, H, W)
            gt_bbox_masks[input_type] = stacked

        chosen_type = random.choice(list(gt_bbox_masks.keys()))
        
        results['gt_bbox_masks'] = gt_bbox_masks[chosen_type]
        results['img_meta']['input_type'] = chosen_type
        return results
