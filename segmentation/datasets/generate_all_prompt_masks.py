import numpy as np
import cv2
import torch
import random
import os
import sys
from tqdm import tqdm
import argparse
import mmcv
from scipy.interpolate import splprep, splev
from mmcv.utils import Config

# --- Import your classes ---
try:
    from class_prompts import (
        GenerateClassUnionBoundingBoxMask,
        GenerateClassCentroidDotMask,
        GenerateClassScribbleMask
    )
except ImportError as e:
    print(f"Error importing from prompts.py: {e}")
    sys.exit(1)

# --- Utility to suppress internal prints ---
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# --- Annotation loader ---
def load_ade20k_annotation(ann_path, ignore_label=255):
    if not os.path.exists(ann_path):
        return None
    gt_semantic_seg = mmcv.imread(ann_path, flag='unchanged', backend='pillow')
    if gt_semantic_seg is None:
        return None
    mask = gt_semantic_seg == 0
    gt_semantic_seg[mask] = ignore_label
    return gt_semantic_seg

# --- Main generation function ---
def generate_masks_for_split(split, base_data_dir, target_classes_set, ignore_label_value,
                              bbox_gen, dot_gen, scribble_gen):
    ann_dir = os.path.join(base_data_dir, 'annotations', split)
    img_dir = os.path.join(base_data_dir, 'images', split)

    out_dir_box = os.path.join(base_data_dir, 'prompt_masks_box', split)
    out_dir_dot = os.path.join(base_data_dir, 'prompt_masks_dot', split)
    out_dir_scribble = os.path.join(base_data_dir, 'prompt_masks_scribble', split)
    os.makedirs(out_dir_box, exist_ok=True)
    os.makedirs(out_dir_dot, exist_ok=True)
    os.makedirs(out_dir_scribble, exist_ok=True)

    annotation_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.png')])

    for ann_filename in tqdm(annotation_files, desc=f"Processing {split}"):
        basename = os.path.splitext(ann_filename)[0]
        ann_path = os.path.join(ann_dir, ann_filename)

        seg_map = load_ade20k_annotation(ann_path, ignore_label=ignore_label_value)
        if seg_map is None:
            continue

        present_labels = np.unique(seg_map)
        current_target_classes_present = [
            label for label in present_labels
            if label in target_classes_set and label != ignore_label_value
        ]

        if not current_target_classes_present:
            continue

        results = {'gt_semantic_seg': seg_map}

        for target_class_id in current_target_classes_present:
            try:
                with suppress_stdout():
                    box_results = bbox_gen(results.copy(), force_class=target_class_id)
                box_mask_tensor = box_results.get('gt_bbox_union_mask')
                if box_mask_tensor is not None:
                    box_mask_np = box_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
                    if box_mask_np.sum() > 0:
                        out_filename = f"{basename}_cls{target_class_id}.npy"
                        np.save(os.path.join(out_dir_box, out_filename), box_mask_np)
            except Exception as e:
                print(f"Error generating BOX mask for {basename} cls {target_class_id}: {e}")

            try:
                with suppress_stdout():
                    dot_results = dot_gen(results.copy(), force_class=target_class_id)
                dot_mask_tensor = dot_results.get('gt_centroid_dot_mask')
                if dot_mask_tensor is not None:
                    dot_mask_np = dot_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
                    if dot_mask_np.sum() > 0:
                        out_filename = f"{basename}_cls{target_class_id}.npy"
                        np.save(os.path.join(out_dir_dot, out_filename), dot_mask_np)
            except Exception as e:
                print(f"Error generating DOT mask for {basename} cls {target_class_id}: {e}")

            try:
                with suppress_stdout():
                    scribble_results = scribble_gen(results.copy(), force_class=target_class_id)
                scribble_mask_tensor = scribble_results.get('gt_scribble_mask')
                if scribble_mask_tensor is not None:
                    scribble_mask_np = scribble_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
                    if scribble_mask_np.sum() > 0:
                        out_filename = f"{basename}_cls{target_class_id}.npy"
                        np.save(os.path.join(out_dir_scribble, out_filename), scribble_mask_np)
            except Exception as e:
                print(f"Error generating SCRIBBLE mask for {basename} cls {target_class_id}: {e}")

# --- Main entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompt masks (box, dot, scribble) for target classes.")
    parser.add_argument(
        '--data_root',
        type=str,
        default='/work3/s203520/advanced_computer_vision/filtered_dataset',
        help='Path to the root directory of the filtered ADE20k dataset.'
    )
    parser.add_argument(
        '--closing_kernel',
        type=int,
        default=7,
        help='Kernel size for morphological closing (e.g., 5, 7). Set to 0 or 1 to disable.'
    )
    parser.add_argument(
        '--scribble_thickness',
        type=int,
        default=2,
        help='Thickness for generated scribble masks.'
    )
    args = parser.parse_args()

    BASE_DATA_DIR = args.data_root
    TARGET_CLASSES = [92,95,146,69,138,116,148,96,122,61,149,142,102,124,94,53,107,105,109,132,85,42,54,60]
    IGNORE_LABEL = 255
    CLOSING_KERNEL = args.closing_kernel if args.closing_kernel > 1 else None
    SCRIBBLE_THICKNESS = args.scribble_thickness

    if not os.path.isdir(BASE_DATA_DIR):
        print(f"Error: Data root directory not found: {BASE_DATA_DIR}")
        sys.exit(1)

    target_classes_set = set(TARGET_CLASSES)

    bbox_gen = GenerateClassUnionBoundingBoxMask(
        target_classes=target_classes_set,
        ignore_label=IGNORE_LABEL,
        closing_kernel_size=CLOSING_KERNEL
    )
    dot_gen = GenerateClassCentroidDotMask(
        target_classes=target_classes_set,
        ignore_label=IGNORE_LABEL,
        closing_kernel_size=CLOSING_KERNEL
    )
    scribble_gen = GenerateClassScribbleMask(
        target_classes=target_classes_set,
        ignore_label=IGNORE_LABEL,
        closing_kernel_size=CLOSING_KERNEL,
        scribble_thickness=SCRIBBLE_THICKNESS,
        min_component_area=20,
        min_scribble_points=3,
        max_scribble_points=10,
        points_per_1k_area=1.5,
        num_interpolated_points=50
    )

    generate_masks_for_split('training', BASE_DATA_DIR, target_classes_set, IGNORE_LABEL,
                             bbox_gen, dot_gen, scribble_gen)
    generate_masks_for_split('validation', BASE_DATA_DIR, target_classes_set, IGNORE_LABEL,
                             bbox_gen, dot_gen, scribble_gen)
