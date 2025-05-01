from builtins import FileNotFoundError
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.ade import ADE20KDataset
import os
import numpy as np
import torch
import mmcv
import json


@DATASETS.register_module()
class CustomDatasetWithClassFilter(ADE20KDataset):
    def __init__(self, class_filter, json_path, **kwargs):
        self.class_filter = class_filter if isinstance(class_filter, list) else [class_filter]
        
        with open(json_path, 'r') as f:
            self.image_classes = json.load(f)
        
        super().__init__(**kwargs)
        
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        data_infos = super().load_annotations(img_dir, img_suffix, ann_dir, seg_map_suffix, split)
        
        filtered_infos = []
        for info in data_infos:
            filename = os.path.basename(info['filename'])
            classes = self.image_classes.get(filename, [])
            if any(cls in classes for cls in self.class_filter):
                filtered_infos.append(info)
        
        print(f'Filtered dataset: {len(filtered_infos)} / {len(data_infos)} images kept.')
        return filtered_infos