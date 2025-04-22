import os, sys

# 1) Add the project root (parent of segmentation/ and vpd/)
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

# 2) Also add the vpd/ directory itself so models_anton.py can be imported directly
vpd_dir = os.path.join(proj_root, "vpd")
sys.path.insert(0, vpd_dir)
# now this works:
from models_anton import UNetWrapper, FrozenCLIPEmbedder
from ldm.models.diffusion.ddpm import LatentDiffusion

#!/usr/bin/env python
import sys, os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import mmcv
from mmseg.datasets import build_dataset, build_dataloader
from mmcv import Config
# ─ add your vpd project to PYTHONPATH so we can import models_anton.py ──────
sys.path.append("/zhome/45/0/155089/adlcv_project/vpd")
from models_anton import UNetWrapper, FrozenCLIPEmbedder
from ldm.models.diffusion.ddpm import LatentDiffusion

# ─────────────────────────────────────────────────────────────────────────────
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
    # simple binary mask per image
    B, C, H, W = img_shape
    mask = torch.zeros((B,1,H,W), device='cpu')
    for i, bbs in enumerate(boxes):
        for x1,y1,x2,y2 in bbs:
            mask[i,0,y1:y2, x1:x2] = 1
    return mask

@torch.no_grad()
def evaluate_iou(model, loader, num_classes, use_boxes=False):
    model.eval()
    tot_inter = np.zeros(num_classes, int)
    tot_union = np.zeros(num_classes, int)

    for batch in tqdm(loader, desc=f"{'with' if use_boxes else 'without'} boxes"):
        imgs  = batch['image'].cuda()            # [B,3,H,W]
        masks = batch['mask'].cuda()             # [B,H,W]
        boxes = batch.get('boxes', None)         # list of lists

        if use_boxes and boxes is not None:
            box_ctrl = make_box_control_feature(imgs.shape, boxes).cuda()
        else:
            box_ctrl = None

        timesteps = torch.zeros(imgs.size(0), dtype=torch.long, device='cuda')
        outputs   = model(x=imgs, timesteps=timesteps,
                          context=None, y=None,
                          box_control=box_ctrl)
        logits = outputs[0]                      # assume [B, num_classes, H, W]
        preds  = logits.argmax(dim=1)            # [B,H,W]

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

    return [
        tot_inter[c]/tot_union[c] if tot_union[c]>0 else float('nan')
        for c in range(num_classes)
    ]

def load_model(ckpt_path, diff_config, device="cuda"):
    # load LDM base
    
    cfg = Config.fromfile(diff_config)
    base = LatentDiffusion(**cfg.model.params)
    sd   = torch.load(ckpt_path, map_location="cpu")
    base.load_state_dict(sd["state_dict"], strict=False)
    base.eval().to(device)

    # 2) wrap it
    wrapper = UNetWrapper(
        unet_a=base,
        interleave_indices=[2,5,8],
        use_attn=True,
        base_size=512,
        attn_selector="up_cross+down_cross"
    ).to(device).eval()

    # 3) optional CLIP (not used here, but available for conditioning)
    clip = FrozenCLIPEmbedder(device=device)
    return wrapper, clip

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--diff-config",  required=True)
    p.add_argument("--mmseg-config", required=True)
    args = p.parse_args()

    # load & wrap diffusion model
    model, clip = load_model(args.ckpt, args.diff_config)

    # build MMseg val dataloader
    cfg = mmcv.Config.fromfile(args.mmseg_config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader  = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False
    )
    num_classes = dataset.num_classes

    # run inference & IoU
    iou_no   = evaluate_iou(model, loader, num_classes, use_boxes=False)
    iou_box  = evaluate_iou(model, loader, num_classes, use_boxes=True)

    # report
    print("\nPer-class IoU (no boxes vs with boxes):")
    for cls in range(num_classes):
        print(f" Class {cls:2d} | {iou_no[cls]:.3f} → {iou_box[cls]:.3f}")
    print(f"\nMean IoU: {np.nanmean(iou_no):.3f} → {np.nanmean(iou_box):.3f}")

if __name__ == "__main__":
    main()



# run this with: 
#python test_boundingboxvpd_on_ade20k.py \
#   --ckpt /work3/s203557/checkpoints/vpd.chkpt \
#   --diff-config /zhome/45/0/155089/adlcv_project/vpd/segmentation/configs/vpd_config.py \
#   --mmseg-config /zhome/45/0/155089/adlcv_project/segmentation/configs/vpd_config.py
