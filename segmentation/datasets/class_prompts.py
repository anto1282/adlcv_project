import numpy as np
import cv2
import torch
import random
from mmseg.datasets.builder import PIPELINES

# #SETTINGS
# # Parameters for generators
# low_iou_classes_zero_indexed = [92,95,146,69,138,116,148,96,122,61,149,142,102,124,94,53,107,105,109,132,85,42,54,60]
# ignore_label = 255

# # --- Instantiate ALL Generators ---
# bbox_generator = GenerateClassUnionBoundingBoxMask(
#     target_classes=low_iou_classes_zero_indexed,
#     ignore_label=ignore_label,
#     closing_kernel_size=KERNEL_SIZE_FOR_CLOSING
# )
# dot_generator = GenerateClassCentroidDotMask(
#     target_classes=low_iou_classes_zero_indexed,
#     ignore_label=ignore_label,
#     closing_kernel_size=3
# )
# scribble_generator = GenerateClassScribbleMask(
#     target_classes=low_iou_classes_zero_indexed,
#     ignore_label=ignore_label,
#     closing_kernel_size=7
#     min_component_area=20,
#     min_scribble_points=3,
#     max_scribble_points=10, # Increased max points slightly
#     points_per_1k_area=1.5,
#     scribble_thickness=2
# )

@PIPELINES.register_module()
class GenerateClassUnionBoundingBoxMask:
    """ Fixed version with force_class and debug prints """
    def __init__(self, target_classes, ignore_label=255, closing_kernel_size=None):
        if not isinstance(target_classes, (list, set, tuple)): raise TypeError("target_classes must be list/set/tuple")
        self.target_classes = set(target_classes)
        self.ignore_label = ignore_label
        self.closing_kernel_size = closing_kernel_size
        if closing_kernel_size is not None:
             if not isinstance(closing_kernel_size, int) or closing_kernel_size <= 1:
                  print(f"Warning: Invalid closing_kernel_size ({closing_kernel_size}). Closing disabled.")
                  self.closing_kernel_size = None
        if not self.target_classes: print("Warning: BBox Generator initialized with empty target_classes.")

    def __call__(self, results, force_class=None): # Added force_class argument
        if 'gt_semantic_seg' not in results: raise KeyError("'gt_semantic_seg' not found.")
        seg_map = results['gt_semantic_seg']
        if isinstance(seg_map, torch.Tensor): seg_map = seg_map.cpu().numpy()
        if seg_map.ndim == 3 and seg_map.shape[0] == 1: seg_map = seg_map.squeeze(0)
        elif seg_map.ndim != 2: raise ValueError(f"Input seg_map shape error: {seg_map.shape}")

        height, width = seg_map.shape
        union_mask = np.zeros((height, width), dtype=np.uint8)
        component_bboxes = []
        # Initialize results keys robustly
        results['selected_prompt_class'] = None
        results['gt_bbox_union_mask'] = torch.from_numpy(union_mask).float().unsqueeze(0)
        results['gt_component_bboxes'] = component_bboxes

        present_labels = np.unique(seg_map)
        available_target_classes = [lbl for lbl in present_labels if lbl in self.target_classes and lbl != self.ignore_label]

        selected_class = None
        if force_class is not None:
            if force_class in available_target_classes:
                selected_class = force_class
                # print(f"[DEBUG BBox] Using forced class: {selected_class}")
            else:
                print(f"[DEBUG BBox] Warning: Forced class {force_class} not in available {available_target_classes}. Skipping.")
                # Return with empty outputs if forced class is invalid
                return results # Already initialized with empty mask/list
        elif available_target_classes:
            selected_class = random.choice(available_target_classes)
            # print(f"[DEBUG BBox] Randomly selected class: {selected_class}")

        if selected_class is None:
            # print("[DEBUG BBox] No target class could be selected.")
            return results # Return with empty outputs

        results['selected_prompt_class'] = selected_class

        class_mask = (seg_map == selected_class).astype(np.uint8)
        processed_mask = class_mask
        if self.closing_kernel_size:
            # print(f"[DEBUG BBox] Applying closing k={self.closing_kernel_size}")
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.closing_kernel_size, self.closing_kernel_size))
            processed_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
        # print(f"[DEBUG BBox] Found {num_labels - 1} component(s) for class {selected_class} AFTER processing.")

        for i in range(1, num_labels):
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(x + w, width), min(y + h, height)
            if x1 < x2 and y1 < y2:
                union_mask[y1:y2, x1:x2] = 1
                component_bboxes.append((x1, y1, x2, y2))

        # print(f"[DEBUG BBox] Collected {len(component_bboxes)} boxes.")
        print(f"[DEBUG BBox] Final BBox mask sum for class {selected_class}: {union_mask.sum()}") # DEBUG SUM
        results['gt_bbox_union_mask'] = torch.from_numpy(union_mask).float().unsqueeze(0)
        results['gt_component_bboxes'] = component_bboxes
        return results
    
# Adds the following keys to the 'results' dictionary:
# - selected_prompt_class (int | None):
#       The integer class ID (0-indexed) that was selected for processing.
#       None if no suitable target class was found.
# - gt_bbox_union_mask (torch.Tensor):
#       Shape: (1, H, W), dtype: float32.
#       A single-channel binary mask where pixels are 1 inside the union of
#       all component bounding boxes for the 'selected_prompt_class', 0 otherwise.
# - gt_component_bboxes (list):
#       A list of tuples, where each tuple is (x1, y1, x2, y2) representing the
#       coordinates of one component's bounding box for the 'selected_prompt_class'.
#       Empty list if no class was selected or no components found.



@PIPELINES.register_module()
class GenerateClassCentroidDotMask:
    """ Fixed version with force_class and debug prints """
    def __init__(self, target_classes, ignore_label=255, closing_kernel_size=None):
        if not isinstance(target_classes, (list, set, tuple)): raise TypeError("target_classes must be list/set/tuple")
        self.target_classes = set(target_classes)
        self.ignore_label = ignore_label
        self.closing_kernel_size = closing_kernel_size
        if closing_kernel_size is not None:
             if not isinstance(closing_kernel_size, int) or closing_kernel_size <= 1:
                  print(f"Warning: Invalid closing_kernel_size ({closing_kernel_size}). Closing disabled.")
                  self.closing_kernel_size = None
        if not self.target_classes: print("Warning: Dot Generator initialized with empty target_classes.")

    def __call__(self, results, force_class=None): # Added force_class argument
        if 'gt_semantic_seg' not in results: raise KeyError("'gt_semantic_seg' not found.")
        seg_map = results['gt_semantic_seg']
        if isinstance(seg_map, torch.Tensor): seg_map = seg_map.cpu().numpy()
        if seg_map.ndim == 3 and seg_map.shape[0] == 1: seg_map = seg_map.squeeze(0)
        elif seg_map.ndim != 2: raise ValueError(f"Input seg_map shape error: {seg_map.shape}")

        height, width = seg_map.shape
        dot_mask = np.zeros((height, width), dtype=np.uint8)
        component_centroids = []
        # Initialize results keys robustly
        # Note: We rely on bbox generator setting selected_prompt_class if run sequentially without force_class
        # It's better practice to reset it here if not forcing, but force_class avoids this issue.
        # results['selected_prompt_class'] = None # Uncomment if running independently
        results['gt_centroid_dot_mask'] = torch.from_numpy(dot_mask).float().unsqueeze(0)
        results['gt_component_centroids'] = component_centroids

        present_labels = np.unique(seg_map)
        available_target_classes = [lbl for lbl in present_labels if lbl in self.target_classes and lbl != self.ignore_label]

        selected_class = None
        if force_class is not None:
            if force_class in available_target_classes:
                selected_class = force_class
                # print(f"[DEBUG Dot] Using forced class: {selected_class}")
            else:
                print(f"[DEBUG Dot] Warning: Forced class {force_class} not in available {available_target_classes}. Skipping.")
                return results # Return with empty outputs
        elif available_target_classes:
            selected_class = random.choice(available_target_classes)
            # print(f"[DEBUG Dot] Randomly selected class: {selected_class}")

        if selected_class is None:
            # print("[DEBUG Dot] No target class could be selected.")
            return results # Return with empty outputs

        # Ensure selected_prompt_class reflects the class actually used
        results['selected_prompt_class'] = selected_class

        class_mask = (seg_map == selected_class).astype(np.uint8)
        processed_mask = class_mask
        if self.closing_kernel_size:
            # print(f"[DEBUG Dot] Applying closing k={self.closing_kernel_size}")
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.closing_kernel_size, self.closing_kernel_size))
            processed_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels_im, stats, centroids_data = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
        # print(f"[DEBUG Dot] Found {num_labels - 1} component(s) for class {selected_class} AFTER processing.")

        for i in range(1, num_labels):
            cx_f, cy_f = centroids_data[i]
            cx, cy = int(round(cx_f)), int(round(cy_f))
            if 0 <= cx < width and 0 <= cy < height:
                dot_mask[cy, cx] = 1
                component_centroids.append((cx, cy))

        # print(f"[DEBUG Dot] Collected {len(component_centroids)} centroids.")
        print(f"[DEBUG Dot] Final Dot mask sum for class {selected_class}: {dot_mask.sum()}") # DEBUG SUM
        results['gt_centroid_dot_mask'] = torch.from_numpy(dot_mask).float().unsqueeze(0)
        results['gt_component_centroids'] = component_centroids
        return results
# Adds the following keys to the 'results' dictionary:
#
# - selected_prompt_class (int | None):
#       The integer class ID (0-indexed) that was selected for processing.
#       None if no suitable target class was found.
# - gt_centroid_dot_mask (torch.Tensor):
#       Shape: (1, H, W), dtype: float32.
#       A single-channel binary mask where single pixels corresponding to the
#       integer coordinates of each component's centroid for the
#       'selected_prompt_class' are set to 1, 0 otherwise.
# - gt_component_centroids (list):
#       A list of tuples, where each tuple is (cx, cy) representing the integer
#       coordinates of one component's centroid for the 'selected_prompt_class'.
#       Empty list if no class was selected or no components found.


@PIPELINES.register_module()
class GenerateClassScribbleMask:
    """ Fixed version with force_class and debug prints """
    def __init__(self, target_classes, ignore_label=255, closing_kernel_size=None,
                 min_component_area=20, min_scribble_points=3, max_scribble_points=8,
                 points_per_1k_area=1.0, num_interpolated_points=50, scribble_thickness=2):

        if not isinstance(target_classes, (list, set, tuple)): raise TypeError("target_classes must be list/set/tuple")
        self.target_classes = set(target_classes)
        self.ignore_label = ignore_label
        self.closing_kernel_size = closing_kernel_size
        if closing_kernel_size is not None:
            if not isinstance(closing_kernel_size, int) or closing_kernel_size <= 1:
                print(f"Warning: Invalid closing_kernel_size ({closing_kernel_size}). Closing disabled.")
                self.closing_kernel_size = None
        self.min_component_area = max(1, min_component_area)
        self.min_scribble_points = max(2, min_scribble_points)
        self.max_scribble_points = max(self.min_scribble_points, max_scribble_points)
        self.points_per_1k_area = max(0, points_per_1k_area)
        self.num_interpolated_points = max(2, num_interpolated_points)
        self.scribble_thickness = max(1, scribble_thickness)
        if not self.target_classes: print("Warning: Scribble Generator initialized with empty target_classes.")

    def __call__(self, results, force_class=None): # Added force_class argument
        if 'gt_semantic_seg' not in results: raise KeyError("'gt_semantic_seg' not found.")
        seg_map = results['gt_semantic_seg']
        if isinstance(seg_map, torch.Tensor): seg_map = seg_map.cpu().numpy()
        if seg_map.ndim == 3 and seg_map.shape[0] == 1: seg_map = seg_map.squeeze(0)
        elif seg_map.ndim != 2: raise ValueError(f"Input seg_map shape error: {seg_map.shape}")

        height, width = seg_map.shape
        scribble_mask = np.zeros((height, width), dtype=np.uint8)
        component_scribble_points = []
        # Initialize results keys robustly
        # results['selected_prompt_class'] = None # See comment in Dot Generator
        results['gt_scribble_mask'] = torch.from_numpy(scribble_mask).float().unsqueeze(0)
        results['gt_component_scribble_points'] = component_scribble_points

        present_labels = np.unique(seg_map)
        available_target_classes = [lbl for lbl in present_labels if lbl in self.target_classes and lbl != self.ignore_label]

        selected_class = None
        if force_class is not None:
            if force_class in available_target_classes:
                selected_class = force_class
                # print(f"[DEBUG Scribble] Using forced class: {selected_class}")
            else:
                print(f"[DEBUG Scribble] Warning: Forced class {force_class} not in available {available_target_classes}. Skipping.")
                return results # Return with empty outputs
        elif available_target_classes:
            selected_class = random.choice(available_target_classes)
            # print(f"[DEBUG Scribble] Randomly selected class: {selected_class}")

        if selected_class is None:
            # print("[DEBUG Scribble] No target class could be selected.")
            return results # Return with empty outputs

        results['selected_prompt_class'] = selected_class

        class_mask = (seg_map == selected_class).astype(np.uint8)
        processed_mask = class_mask
        if self.closing_kernel_size:
            # print(f"[DEBUG Scribble] Applying closing k={self.closing_kernel_size}")
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.closing_kernel_size, self.closing_kernel_size))
            processed_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
        # print(f"[DEBUG Scribble] Found {num_labels - 1} component(s) for class {selected_class} AFTER processing.")

        for i in range(1, num_labels):
            component_area = stats[i, cv2.CC_STAT_AREA]
            if component_area < self.min_component_area: continue

            points_based_on_area = int(round(self.points_per_1k_area * (component_area / 1000.0)))
            num_points = np.clip(self.min_scribble_points + points_based_on_area, self.min_scribble_points, self.max_scribble_points)

            rows, cols = np.where(labels_im == i)
            if len(rows) < num_points: num_points = len(rows)
            if num_points < 2: continue

            sampled_indices = random.sample(range(len(rows)), num_points)
            sampled_points = [(cols[idx], rows[idx]) for idx in sampled_indices]
            sampled_points_np = np.array(sampled_points)
            component_scribble_points.append(sampled_points) # Store list of tuples

            try:
                if num_points >= 3:
                    x_coords, y_coords = sampled_points_np[:, 0], sampled_points_np[:, 1]
                    tck, u = splprep([x_coords, y_coords], s=0, k=min(3, num_points - 1))
                    u_fine = np.linspace(0, 1, self.num_interpolated_points)
                    x_fine, y_fine = splev(u_fine, tck)
                    points_to_draw = np.vstack((x_fine, y_fine)).T.round().astype(np.int32)
                    cv2.polylines(scribble_mask, [points_to_draw], isClosed=False, color=1, thickness=self.scribble_thickness, lineType=cv2.LINE_AA)
                elif num_points == 2:
                    pt1 = tuple(sampled_points_np[0].astype(int))
                    pt2 = tuple(sampled_points_np[1].astype(int))
                    cv2.line(scribble_mask, pt1, pt2, color=1, thickness=self.scribble_thickness, lineType=cv2.LINE_AA)
            except Exception as e:
                 print(f"[DEBUG Scribble] Error generating spline/line for component {i}: {e}. Sampled points: {sampled_points}")

        # print(f"[DEBUG Scribble] Collected points for {len(component_scribble_points)} scribbles.")
        print(f"[DEBUG Scribble] Final Scribble mask sum for class {selected_class}: {scribble_mask.sum()}") # DEBUG SUM
        results['gt_scribble_mask'] = torch.from_numpy(scribble_mask).float().unsqueeze(0)
        results['gt_component_scribble_points'] = component_scribble_points
        return results
# Adds the following keys to the 'results' dictionary:
#
# - selected_prompt_class (int | None):
#       The integer class ID (0-indexed) that was selected for processing.
#       None if no suitable target class was found.
# - gt_scribble_mask (torch.Tensor):
#       Shape: (1, H, W), dtype: float32.
#       A single-channel binary mask where pixels along the generated scribble
#       lines (based on spline/line interpolation) for all components of the
#       'selected_prompt_class' are set to 1, 0 otherwise. Thickness depends
#       on the 'scribble_thickness' parameter.
# - gt_component_scribble_points (list):
#       A list where each element corresponds to one generated scribble. Each
#       element is itself a list of tuples `[(x1, y1), (x2, y2), ...]`,
#       representing the coordinates of the points sampled within that component
#       used to generate the scribble spline/line. Empty list if no class was
#       selected or no suitable components found.




