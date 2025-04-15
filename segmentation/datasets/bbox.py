import numpy as np
import cv2
import torch
from mmseg.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer

@PIPELINES.register_module()
class GenerateBoundingBoxMasksFromSeg:
    def __init__(self, ignore_label=255):
        self.ignore_label = ignore_label

    def __call__(self, results):
        seg_map = results['gt_semantic_seg']
        height, width = seg_map.shape
        bbox_masks = []

        for label in np.unique(seg_map):
            if label == self.ignore_label:
                continue
            mask = (seg_map == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                bbox_mask = np.zeros((height, width), dtype=np.uint8)
                bbox_mask[y:y + h, x:x + w] = 1
                bbox_masks.append(torch.from_numpy(bbox_mask))  # Convert to Tensor 👈

        results['gt_bbox_masks'] = DataContainer(bbox_masks, stack=False)  # ✅ Proper container
        return results