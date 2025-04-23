#!/usr/bin/env python
import os, sys

# this script lives in .../Run_inference_on_VPDwControlnet/
root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root)         # lets us do `import hooks` and `import vpd`


# ─────────────────────────────────────────────────────────────────────────────
# 1) Ensure Python can import your two top‐level packages:
this_dir    = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, '..', '..'))
sys.path.insert(0, project_root)                     # for `import segmentation`
sys.path.insert(0, os.path.join(project_root, 'vpd'))  # for `import models_anton`
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import torch
import numpy as np
from mmcv import Config
from mmengine.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmcv.parallel import collate
from mmseg.datasets import build_dataset, build_dataloader

from models_anton import UNetWrapper, FrozenCLIPEmbedder
from ldm.models.diffusion.ddpm import LatentDiffusion
from tqdm import tqdm
import os, sys
from mmcv import Config
from mmseg.apis import init_segmentor, single_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint
import torch

import os, sys
root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root)                            # for hooks/, vpd/
sys.path.insert(0, os.path.abspath(os.path.join(root, '..',)))    # for segmentation/
sys.path.insert(0, os.path.join(root, 'vpd'))       # for vpd/models_anton.py

# ─────────────────────────────────────────────────────────────────────────────
#  1) Now import the package that registers VPDSeg:
import Run_inference_on_VPDwControlnet.models      # <— this will execute segmentation/models/__init__.py
#  2) Also import your diffusion-wrapper code
import vpd.models_anton         # <— this registers the UNetWrapper etc.
# ── 2) Helpers ───────────────────────────────────────────────────────────────


def compute_class_iou(pred_flat, tgt_flat, num_classes):
    ious = []
    for cls in range(num_classes):
        p = (pred_flat == cls)
        t = (tgt_flat  == cls)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        ious.append(inter/union if union>0 else float('nan'))
    return ious

def make_box_control_feature(img_shape, boxes):
    """Turn list of bounding boxes into a single‐channel mask."""
    B, C, H, W = img_shape
    mask = torch.zeros((B,1,H,W), dtype=torch.float32)
    for i, bbs in enumerate(boxes):
        for x1,y1,x2,y2 in bbs:
            mask[i,0,y1:y2, x1:x2] = 1.0
    return mask

@torch.no_grad()
def evaluate_iou(model, loader, num_classes, use_boxes=False):
    model.eval()
    tot_inter = np.zeros(num_classes, int)
    tot_union = np.zeros(num_classes, int)

    for batch in tqdm(loader, desc=f"{'with' if use_boxes else 'without'} boxes"):
        if isinstance(batch, list):
            batch = batch[0]  # Unwrap the list if it's a single item batch
       
        print(batch.keys())
        imgs  = batch['img'].cuda()            # [B,3,H,W]
        masks = batch['gt_semantic_seg'].squeeze(1).cuda()  # [B,H,W]
        boxes = batch['gt_bbox_masks'].cuda()         # list of lists

        if use_boxes:
            box_ctrl = make_box_control_feature(imgs.shape, boxes).cuda()
        else:
            box_ctrl = None

        # one‐step “inference”: treat timesteps=0, no text conditioning
        timesteps = torch.zeros(imgs.size(0), dtype=torch.long, device='cuda')
        outputs   = model(
            x=imgs,
            timesteps=timesteps,
            context=None,
            y=None,
            box_control=box_ctrl
        )
        # assume first output is the logits for segmentation
        logits = outputs[0]        # [B, num_classes, H, W]
        preds  = logits.argmax(dim=1)

        for b in range(preds.size(0)):
            pf = preds[b].view(-1)
            tf = masks[b].view(-1)
            ious = compute_class_iou(pf, tf, num_classes)
            for cls, iou in enumerate(ious):
                if not np.isnan(iou):
                    inter = ((pf==cls)&(tf==cls)).sum().item()
                    union = ((pf==cls)|(tf==cls)).sum().item()
                    tot_inter[cls] += inter
                    tot_union[cls] += union

    return [tot_inter[c]/tot_union[c] if tot_union[c]>0 else float('nan')
            for c in range(num_classes)]

# ── 3) Model loader ───────────────────────────────────────────────────────────

def load_vpd_model(ckpt_path, cfg_path, device='cuda'):
    # 1) Load the MMCV config
    cfg = Config.fromfile(cfg_path)

    # 2) Build the VPDSeg segmentor 
    #    Note: train_cfg=None because we only want inference behavior
    model = build_segmentor(
        cfg.model,
        train_cfg=None,
        test_cfg=None  # or {} if none
    )

    # 3) Load the checkpoint (fills in both diffusion & decode_head weights)
    load_checkpoint(model, ckpt_path, map_location='cpu')

    # 4) Move to GPU and eval
    model.eval().to(device)
    return model, cfg

# ── 4) Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) load config
    cfg = Config.fromfile("configs/vpd_config.py")

    # 2) we want to run on validation (so we have GT masks)
    cfg.data.test = cfg.data.val
    cfg.data.test.test_mode = True

    # 3) init model (this will register your VPDSeg, hooks, etc.)
    model = init_segmentor(
        cfg,
        checkpoint=None,
        device="cuda:0"
        #cfg_options=dict()  # any override, e.g. {'load_from': ckpt_path}
    )

    # 4) load your trained weights
    load_checkpoint(model, "/work3/s203557/checkpoints/vpd.chkpt", map_location="cpu")

    model.cuda()
    model.eval()

    # 5) build the dataloader
    test_dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # 6) run inference + metric
    # this uses cfg.evaluation = dict(metric='mIoU') by default
    results = single_gpu_test(model, data_loader, show=False)

    # 7) print out the evaluation results
    # this will read your `evaluation` section (mIoU, per-class IoU, etc.)
    eval_res = test_dataset.evaluate(results, **cfg.evaluation)
    print("\nFinal evaluation:")
    for k, v in eval_res.items():
        print(f"  {k}: {v:.4f}")


