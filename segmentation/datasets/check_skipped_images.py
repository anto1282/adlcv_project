import numpy as np
import os
import random
import mmcv
import sys

# --- Configuration (Match the generation script!) ---
BASE_DATA_DIR = '/work3/s203520/advanced_computer_vision/filtered_dataset'
# IMPORTANT: Use the *exact* list that the generation script used
TARGET_CLASSES_SET = set([92,95,146,69,138,116,148,96,122,61,149,142,102,124,94,53,107,105,109,132,85,42,54,60])
IGNORE_LABEL = 255
SPLIT = 'validation'
NUM_IMAGES_TO_CHECK = 800 # Check a few random images

# --- Re-use or copy the loading function ---
def load_ade20k_annotation(ann_path, ignore_label=255):
    if not os.path.exists(ann_path): return None
    gt_semantic_seg = mmcv.imread(ann_path, flag='unchanged', backend='pillow')
    if gt_semantic_seg is None: return None
    mask = gt_semantic_seg == 0
    gt_semantic_seg = gt_semantic_seg
    gt_semantic_seg[mask] = ignore_label
    return gt_semantic_seg

# --- Main Check Logic ---
if __name__ == "__main__":
    ann_dir = os.path.join(BASE_DATA_DIR, 'annotations', SPLIT)
    if not os.path.isdir(ann_dir):
        print(f"ERROR: Annotation directory not found: {ann_dir}"); sys.exit(1)

    print(f"--- Verifying Labels in {SPLIT} Split ---")
    print(f"Target Classes Set (as used in script): {sorted(list(TARGET_CLASSES_SET))}")
    print(f"Ignore Label: {IGNORE_LABEL}")
    print("-" * 40)

    try:
        annotation_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.png')])
        if not annotation_files: raise FileNotFoundError(f"No PNGs in {ann_dir}")
    except Exception as e:
        print(f"Error listing annotations: {e}"); sys.exit(1)

    # Sample some files to check
    files_to_check = random.sample(annotation_files, min(NUM_IMAGES_TO_CHECK, len(annotation_files)))

    skipped_count = 0
    found_count = 0

    for ann_filename in files_to_check:
        ann_path = os.path.join(ann_dir, ann_filename)
        seg_map = load_ade20k_annotation(ann_path, ignore_label=IGNORE_LABEL)

        if seg_map is None:
            print(f"File: {ann_filename} -> ERROR Loading Annotation")
            continue

        present_labels = np.unique(seg_map)
        # Find intersection AFTER loading/processing
        current_target_classes_present = [
            label for label in present_labels
            if label in TARGET_CLASSES_SET and label != IGNORE_LABEL
        ]

        print(f"File: {ann_filename}")
        print(f"  Present Labels (0-indexed + ignore): {sorted(list(present_labels))}")
        if not current_target_classes_present:
            print(f"  MATCH CHECK: NO target classes found in TARGET_CLASSES_SET.")
            skipped_count += 1
        else:
            print(f"  MATCH CHECK: FOUND target classes: {sorted(current_target_classes_present)}")
            found_count += 1
        print("-" * 20)

    print("\n--- Verification Summary ---")
    print(f"Checked {len(files_to_check)} random samples.")
    print(f"Samples where target classes WERE found: {found_count}")
    print(f"Samples where target classes WERE NOT found (would be skipped): {skipped_count}")

    if skipped_count > 0 and found_count > 0:
         print("\nConclusion: It seems some images in the 'filtered' dataset DO contain")
         print("target classes, while others DO NOT (after 0-based label conversion).")
         print("Verify if TARGET_CLASSES_SET uses the correct 0-based indices (0-149).")
         print("If the indices are correct, the original filtering might have included")
         print("images that don't actually contain the target classes.")
    elif skipped_count == len(files_to_check) and skipped_count > 0:
         print("\nConclusion: NONE of the checked samples contained the target classes")
         print("after 0-based label conversion. This STRONGLY suggests your")
         print(f"TARGET_CLASSES_SET {sorted(list(TARGET_CLASSES_SET))} uses 1-based indexing,")
         print("but the loaded maps use 0-based indexing (0-149).")
         print("You likely need to subtract 1 from each ID in your target list.")
    elif found_count == len(files_to_check):
         print("\nConclusion: All checked samples contained target classes. The discrepancy")
         print("might be elsewhere or related to specific images not checked.")
         print("Consider increasing NUM_IMAGES_TO_CHECK or checking specific examples.")