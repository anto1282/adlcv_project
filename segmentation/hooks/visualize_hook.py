from mmcv.runner import HOOKS, Hook
import os
import mmcv
import torch
import numpy as np
import cv2
import random
ADE20K_palette = [
    [120, 120, 120], [180, 120, 120], [  6, 230, 230], [ 80,  50,  50],
    [  4, 200,   3], [120, 120,  80], [140, 140, 140], [204,   5, 255],
    [230, 230, 230], [  4, 250,   7], [224,   5, 255], [235, 255,   7],
    [150,   5,  61], [120, 120,  70], [  8, 255,  51], [255,   6,  82],
    [143, 255, 140], [204, 255,   4], [255,  51,   7], [204,  70,   3],
    [  0, 102, 200], [ 61, 230, 250], [255,   6,  51], [ 11, 102, 255],
    [255,   7,  71], [255,   9, 224], [  9,   7, 230], [220, 220, 220],
    [255,   9,  92], [112,   9, 255], [  8, 255, 214], [  7, 255, 224],
    [255, 184,   6], [ 10, 255,  71], [255,  41,  10], [  7, 255, 255],
    [224, 255,   8], [102,   8, 255], [255,  61,   6], [255, 194,   7],
    [255, 122,   8], [  0, 255,  20], [255,   8,  41], [255,   5, 153],
    [  6,  51, 255], [235,  12, 255], [160, 150,  20], [  0, 163, 255],
    [140, 140, 140], [250,  10,  15], [ 20, 255,   0], [ 31, 255,  12],
    [255,  31,   0], [255, 224,   0], [153, 255,   0], [  0,  0, 255],
    [255,  71,   0], [  0, 235, 255], [  0, 173, 255], [ 31,   0, 255],
    [ 11, 200, 200], [255,  82,   0], [  0, 255, 245], [  0,  61, 255],
    [  0, 255, 112], [  0, 255, 133], [255,  94,   0], [  0, 224, 255],
    [  0, 255, 255], [255, 224,   0], [255, 153,   0], [255, 112,   0],
    [255, 191,   0], [255,  0,   0], [255,  72,   0], [255, 255,   0],
    [255, 136,   0], [255, 255, 224], [255, 255,   0], [  0, 255,   0],
    [  0, 255, 127], [  0, 255, 255], [255,   0, 255], [ 80, 200, 120],
    [255, 165,   0], [173, 255,  47], [255,  20, 147], [255, 105, 180],
    [255, 182, 193], [138,  43, 226], [139,   0, 139], [ 75,   0, 130],
    [106,  90, 205], [ 72,  61, 139], [199,  21, 133], [255,  69,   0],
    [124, 252,   0], [127, 255,   0], [  0, 250, 154], [173, 255,  47],
    [  0, 255, 127], [ 60, 179, 113], [ 46, 139,  87], [  0, 100,   0],
    [ 34, 139,  34], [  0, 128,   0], [107, 142,  35], [154, 205,  50],
    [ 85, 107,  47], [  0, 255, 255], [ 64, 224, 208], [ 72, 209, 204],
    [  0, 206, 209], [ 32, 178, 170], [ 47,  79,  79], [112, 128, 144],
    [119, 136, 153], [105, 105, 105], [128, 128, 128], [192, 192, 192],
    [211, 211, 211], [220, 220, 220], [169, 169, 169], [255, 250, 250],
    [240, 255, 255], [240, 248, 255], [245, 245, 245], [255, 255, 255]
]

@HOOKS.register_module()
class TrainVisualizeHook(Hook):
    def __init__(self, interval=1000, num_samples=4, save_dir='train_vis'):
        self.interval = interval
        self.num_samples = num_samples
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def denormalize_img(self,img_tensor):
        """Reverses (img - mean) / std normalization"""
        mean = np.array([127.5, 127.5, 127.5]).reshape(3, 1, 1)
        std = np.array([127.5, 127.5, 127.5]).reshape(3, 1, 1)
        img = img_tensor.cpu().numpy() * std + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    def apply_palette(self, mask, palette):
        """Convert a mask with class indices to RGB using a palette."""
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_mask[mask == label] = color
        return color_mask
    
    def after_train_iter(self, runner):
        if runner.iter % self.interval != 0:
            return

        # Get the real PyTorch DataLoader
        dataloader = runner.data_loader._dataloader
        data_iter = iter(dataloader)
        batch = next(data_iter)

        imgs = batch['img'].data[0]
        gts = batch['gt_semantic_seg'].data[0].cpu().numpy()
        boxes = batch.get('gt_bbox_masks', None)

        # Run inference
        runner.model.eval()
        device = next(runner.model.module.parameters()).device 
        with torch.no_grad():
            preds = runner.model.module.simple_test(
                batch['img'].data[0].to(device),
                batch['img_metas'].data[0],
                gt_bbox_masks=boxes.data[0].to(device) if boxes is not None else None
            )
        runner.model.train()

        for i in range(min(self.num_samples, len(imgs))):
            img = imgs[i]
            img = self.denormalize_img(img).transpose(1, 2, 0).astype(np.uint8)
            pred = preds[i].astype(np.uint8)
            gt = gts[i].squeeze().astype(np.uint8)
            vis_img = img.copy()
            # flip = batch["img_metas"].data[0][i].get("flip",False)
            # flip_direction = batch["img_metas"].data[0][i].get("flip_direction", "horizontal")
            # # --- reverse flip ---
            # if flip:
            #     if flip_direction == 'horizontal':
            #         vis_img = np.fliplr(vis_img)
            #         gt = np.fliplr(gt)
            #         pred = np.fliplr(pred)
            #     elif flip_direction == 'vertical':
            #         vis_img = np.flipud(vis_img)
            #         gt = np.flipud(gt)
            #         pred = np.flipud(pred)
            # Draw bounding boxes if available
            if boxes is not None:
                # boxes: [C, H, W]ø
                box_tensor = boxes.data[i].cpu().numpy().astype(np.uint8)
                vis_img = np.ascontiguousarray(vis_img)
                count = 0
                for c in range(box_tensor.shape[0]):
                    box_mask = box_tensor[c]

                    if box_mask.sum() == 0:
                        count += 1
                        continue  # skip empty masks
                    if box_mask.ndim == 3:
                        box_mask = box_mask.squeeze()

                    ys, xs = np.where(box_mask)
                    y1, y2 = ys.min(), ys.max()
                    x1, x2 = xs.min(), xs.max()

                    color = tuple(random.randint(0, 255) for _ in range(3))
                    vis_img = cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                print(f"Empty masks = {count}")
            colored_pred = self.apply_palette(pred, ADE20K_palette)
            colored_gt = self.apply_palette(gt, ADE20K_palette)

            mmcv.imwrite(vis_img, os.path.join(self.save_dir, f'{runner.iter}_img_{i}.png'))
            mmcv.imwrite(colored_gt, os.path.join(self.save_dir, f'{runner.iter}_gt_{i}.png'))
            mmcv.imwrite(colored_pred, os.path.join(self.save_dir, f'{runner.iter}_pred_{i}.png'))

