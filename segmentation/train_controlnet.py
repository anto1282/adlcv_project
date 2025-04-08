import torch
import torch.nn.functional as F
import mmseg.models.decode_heads.decode_head as decode_head_module

import traceback

# def safe_resize(input,
#                 size=None,
#                 scale_factor=None,
#                 mode='nearest',
#                 align_corners=None,
#                 warning=True):
#     if isinstance(size, torch.Size):
#         size = tuple(size)
#     elif isinstance(size, int):
#         size = (size, size)
#     elif isinstance(size, (list, tuple)) and len(size) == 1:
#         print(f"\n[💥 BAD SIZE DETECTED] size={size}, converting to {(size[0], size[0])}")
#         traceback.print_stack(limit=5)  # show where this came from
#         size = (size[0], size[0])
#     elif isinstance(size, (list, tuple)) and len(size) > 2:
#         size = tuple(size[:2])

#     print(f"[DEBUG resize] input.shape={input.shape}, size={size}")
#     return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


# # ⚠️ Patch the resize used directly in decode_head.py
# decode_head_module.resize = safe_resize
# print("[INFO] decode_head.resize monkey-patched ✅")


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




# --- Configs ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2
NUM_EPOCHS = 5
LR = 1e-4

# --- Dataset & Dataloader ---
IMG_SIZE = (512, 512)  # or 256x256, 384x384, etc.

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])
train_dataset = VOCDatasetWithBBoxes(root_dir='/work3/s203557/data/VOCdevkit/VOC2012', split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# --- Model ---
model = VPDSeg(
    decode_head=dict(
        type='FCNHead',
        in_channels=1280,  # adjust based on your unet wrapper output
        channels=128,
        num_convs=2,
        kernel_size=3,
        num_classes=21,  # for VOC
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
    ),
    sd_path='/work3/s203557/checkpoints/v1-5-pruned-emaonly.ckpt',
    sd_config='/zhome/b6/d/154958/ADLCV_Project/VPD/stable-diffusion/configs/stable-diffusion/v1-inference.yaml',
    class_embedding_path='./segmentation/class_embeddings.pth'
)

checkpoint = torch.load("/work3/s203557/checkpoints/vpd.chkpt", map_location=DEVICE)
model.load_state_dict(checkpoint, strict=False)  # allow partial load if needed
print("✅ Loaded pretrained VPD weights")




model.to(DEVICE)
model.train()


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    epoch_loss = 0.0

    for imgs, boxes, segs in pbar:
        imgs = torch.stack(imgs).to(DEVICE)  
        print(imgs.shape) 
        H, W = imgs.shape[2], imgs.shape[3]

        gt_segs = torch.stack([torch.tensor(seg, dtype=torch.long) for seg in segs]).to(DEVICE)
        batch_boxes = [b.to(DEVICE) for b in boxes]

        optimizer.zero_grad()
        img_metas = [
        dict(
            ori_shape=(int(H), int(W)),
            img_shape=(int(H), int(W)),
            scale_factor=(1.0, 1.0),
            flip=False,
            flip_direction='horizontal'
        )
        for _ in range(imgs.shape[0])
    ]
        
        gt_segs = gt_segs.unsqueeze(1)
        print("🌀 Features from UNetWrapper to decode head:", 
            [f.shape for f in imgs] if isinstance(imgs, list) else imgs.shape)
        loss_dict = model.forward_train(imgs, img_metas = img_metas,
                                        gt_semantic_seg=gt_segs, gt_bboxes=batch_boxes)
        loss = sum(loss_dict.values())
        for name, param in model.control_net.named_parameters():
            if param.grad is None:
                print(f"[❌ NO GRAD] {name}")
            else:
                print(f"[✅ GRAD] {name} — grad.norm = {param.grad.norm():.4f}")
        loss.backward()


        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")
