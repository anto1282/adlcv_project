import torch.utils.checkpoint
import numpy as np
def safe_no_checkpoint(fn, inputs, params=None, flag=None):
    if isinstance(inputs, tuple):
        return fn(*inputs)
    else:
        return fn(inputs)

torch.utils.checkpoint.checkpoint = safe_no_checkpoint
print("[🚫] Overrode torch.utils.checkpoint.checkpoint globally")

import ldm.modules.diffusionmodules.util as diffusion_utils

def safe_checkpoint(run_function, inputs, params, flag):
    # Only checkpoint if any parameter requires gradients
    requires_grad = any(p.requires_grad for p in params)
    if not flag or not requires_grad:
        return run_function(*inputs)
    else:
        return diffusion_utils.CheckpointFunction.apply(run_function, len(inputs), *(inputs + tuple(params)))

diffusion_utils.checkpoint = safe_checkpoint
print("[✅] Monkey-patched checkpoint with gradient-safe fallback")


import functools
import torch
import torch.nn.functional as F
import mmseg.models.decode_heads.decode_head as decode_head_module
import types

from torchvision import transforms
from segmentation.dataloader.voc_dataloader import VOCDatasetWithBBoxes


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from mmcv import Config

# Replace with your custom dataset
from segmentation.models.vpd_seg import VPDSeg

import torch.nn.functional as F


import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import os
from mmcv import ConfigDict

def draw_boxes_to_mask(bboxes, size):
    mask = torch.zeros(size, dtype=torch.uint8)
    for box in bboxes:
        x1, y1, x2, y2 = box.int()
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, size[1] - 1)
        y2 = min(y2, size[0] - 1)
        mask[y1:y2, x1:x2] = 1
    return mask.unsqueeze(0)  # shape [1, H, W]

def visualize_debug(img, bboxes, gt_mask, pred_mask, epoch, sample_idx=0, save_dir="debug_vis", model=None, img_meta=None):
    os.makedirs(save_dir, exist_ok=True)
    img = img.squeeze(0)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    gt_np = gt_mask.squeeze().cpu().numpy()
    pred_np = pred_mask.cpu().numpy()
    box_mask = draw_boxes_to_mask(bboxes, size=gt_mask.shape[1:])  # shape [1, H, W]
    box_np = box_mask.squeeze(0).cpu().numpy()

    # NEW: predict without ControlNet (no bounding boxes)
    if model is not None and img_meta is not None:
        with torch.no_grad():
            pred_noctrl = model.simple_test(img.unsqueeze(0), img_meta=[img_meta], gt_bboxes=None, rescale=True)
            pred_noctrl_np = torch.tensor(pred_noctrl[0]).cpu().numpy()
    else:
        pred_noctrl_np = np.zeros_like(pred_np)

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    axs[0].imshow(img_np)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(box_np, cmap="Greens")
    axs[1].set_title("Box Mask")
    axs[1].axis("off")

    axs[2].imshow(pred_np, cmap="jet")
    axs[2].set_title("Prediction (w/ Box)")
    axs[2].axis("off")

    axs[3].imshow(pred_noctrl_np, cmap="jet")
    axs[3].set_title("Prediction (no Box)")
    axs[3].axis("off")

    axs[4].imshow(gt_np, cmap="tab20")
    axs[4].set_title("Ground Truth")
    axs[4].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_sample_{sample_idx}.png"))
    plt.close()


# --- Configs ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = "cpu"
print("Training on device",DEVICE)
BATCH_SIZE = 3
NUM_EPOCHS = 10
LR = 1e-6

# --- Dataset & Dataloader ---
IMG_SIZE = (512, 512)  # or 256x256, 384x384, etc.

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])
train_dataset = VOCDatasetWithBBoxes(root_dir='/work3/s203557/data/VOCdevkit/VOC2012', split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# --- Model ---
model = VPDSeg(base_size = IMG_SIZE[0],
    neck = dict(
        type='FPN',
        in_channels=[320, 661, 1301, 1280], # Skal være output a unet + antal klasser
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,  # adjust based on your unet wrapper output
        in_index=0,
        channels=256,
        num_convs=2,
        num_classes=21,  # for VOC
        concat_input = True,
        dropout_ratio=0.1,
        align_corners=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
    ),
    sd_path='/work3/s203557/checkpoints/v1-5-pruned-emaonly.ckpt',
    sd_config='/zhome/b6/d/154958/ADLCV_Project/VPD/stable-diffusion/configs/stable-diffusion/v1-inference.yaml',
    class_embedding_path='./segmentation/voc2012_class_embeddings.pt',
    test_cfg=ConfigDict(dict(mode='whole'))

)

for name, param in model.named_parameters():
    param.requires_grad = False

for name,param in model.decode_head.named_parameters():
    param.requires_grad = True

for name, param in model.text_adapter.named_parameters():
    param.requires_grad = True

for name, param in model.box_encoder.named_parameters():
    param.requires_grad = True
model.gamma.requires_grad = True




checkpoint = torch.load("/work3/s203557/checkpoints/vpd.chkpt", map_location=DEVICE)
model.load_state_dict(checkpoint, strict=False)  # allow partial load if needed
print("✅ Loaded pretrained VPD weights")

model.to(DEVICE)
model.train()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)


# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    epoch_loss = 0.0
    visualized = False

    if epoch == 3:
        for name, param in model.unet.trainable_unet.named_parameters():
            param.requires_grad = True
            
        for param in model.unet.zero_convs.parameters():
            param.requires_grad = True


    for batch_idx, (imgs, boxes, segs) in enumerate(pbar):
        imgs = torch.stack(imgs).to(DEVICE)  
        H, W = imgs.shape[2], imgs.shape[3]

        gt_segs = torch.stack([torch.tensor(seg, dtype=torch.long) for seg in segs]).to(DEVICE)
        gt_segs = gt_segs.unsqueeze(1)
        batch_boxes = [b.to(DEVICE) for b in boxes]

        optimizer.zero_grad()
        img_metas = [
            dict(
                ori_shape=(int(H), int(W)),
                img_shape=(int(H), int(W)),
                scale_factor=(1.0, 1.0),
                flip=True,
                flip_direction='horizontal',

            )
            for _ in range(imgs.shape[0])
        ]
        if visualized == False:
            model.eval()
            with torch.no_grad():
                for i in range(2):  # visualize first 2 samples in the batch
                    img = imgs[i].unsqueeze(0)
                    gt = gt_segs[i]
                    box_list = batch_boxes[i].unsqueeze(0)  # 🆕 clearer name
                    pred_logits = model.simple_test(
                        img,
                        img_meta=[img_metas[i]],
                        gt_bboxes=box_list,
                        rescale=False
                    )
                    pred_mask = torch.tensor(pred_logits[0]).to(DEVICE)  # [H, W]

                    visualize_debug(
                        img=img,
                        bboxes=box_list.squeeze(),
                        gt_mask=gt,
                        pred_mask=pred_mask,
                        epoch=epoch,
                        sample_idx=i,
                        save_dir="debug_vis",
                        model=model,                # pass model here
                        img_meta=img_metas[i]       # and its metadata
                    )
            model.train()
            visualized = True

        if epoch > 2:
            loss_dict = model.forward_train(imgs, img_metas=img_metas,
                                            gt_semantic_seg=gt_segs, gt_bboxes=batch_boxes)
        else: 
            loss_dict = model.forward_train(imgs, img_metas=img_metas,
                                            gt_semantic_seg=gt_segs, gt_bboxes=batch_boxes)
            
        
        loss = sum(loss_dict.values())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), gamma = model.gamma.mean.item)

    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")
