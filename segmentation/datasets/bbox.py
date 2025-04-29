import numpy as np
import cv2
import torch
import random
from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class GenerateBoundingBoxMasksFromSeg:
    def __init__(self, ignore_label=255, max_boxes=8, min_area=100, max_area_ratio=0.3):
        self.ignore_label = ignore_label
        self.max_boxes = max_boxes
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    def __call__(self, results):
        seg_map = results['gt_semantic_seg']
        height, width = seg_map.shape
        masks = np.zeros((self.max_boxes, height, width), dtype=np.uint8)

        candidate_boxes = []

        for label in np.unique(seg_map):
            if label == self.ignore_label:
                continue
            mask = (seg_map == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area >= self.min_area and area <= self.max_area_ratio * height * width:
                    candidate_boxes.append((x, y, x + w, y + h))

        random.shuffle(candidate_boxes)
        for i, (x1, y1, x2, y2) in enumerate(candidate_boxes[:self.max_boxes]):
            masks[i, y1:y2, x1:x2] = 1

        results['gt_bbox_masks'] = torch.from_numpy(masks).float()
        return results