import torch.utils.checkpoint

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

def visualize_prediction(image, box_map, pred_mask, gt_mask=None, step=None, path="plots" ):
    """
    Visualizes input image, control box map, prediction, and optionally ground truth.
    image: Tensor [C, H, W]
    box_map: Tensor [1, H, W]
    pred_mask: Tensor [1, H, W]
    gt_mask: Tensor [1, H, W] (optional)
    """

    def to_np(tensor):
        return tensor.detach().cpu().numpy()

    image_np = to_np(image).transpose(1, 2, 0)
    box_np = to_np(box_map.squeeze(0))
    pred_np = to_np(pred_mask.squeeze(0))
    gt_np = to_np(gt_mask.squeeze(0)) if gt_mask is not None else None

    num_plots = 3 if gt_mask is None else 4
    fig, axs = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))

    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(box_np, cmap='Greens')
    axs[1].set_title("Bounding Box Map")
    axs[1].axis('off')

    axs[2].imshow(pred_np, cmap='jet')
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    if gt_mask is not None:
        axs[3].imshow(gt_np, cmap='gray')
        axs[3].set_title("Ground Truth")
        axs[3].axis('off')

    if step is not None:
        fig.suptitle(f"Epoch {step}", fontsize=16)


    plt.tight_layout()
    os.makedirs(path, exist_ok=True)

    plt.savefig(f"{path}/pred_plot{step}.png")


# --- Configs ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on device",DEVICE)
BATCH_SIZE = 5
NUM_EPOCHS = 10
LR = 1e-5

# --- Dataset & Dataloader ---
IMG_SIZE = (256, 256)  # or 256x256, 384x384, etc.

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])
train_dataset = VOCDatasetWithBBoxes(root_dir='/work3/s203557/data/VOCdevkit/VOC2012', split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# --- Model ---
model = VPDSeg(base_size = IMG_SIZE[0],
    decode_head=dict(
        type='FCNHead',
        in_channels=320,  # adjust based on your unet wrapper output
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
    class_embedding_path='./segmentation/class_embeddings.pth',
    test_cfg=ConfigDict(dict(mode='whole'))

)

for name, param in model.named_parameters():
    param.requires_grad = False

for name, param in model.box_encoder.named_parameters():
    param.requires_grad = True

for name, param in model.unet.trainable_unet.named_parameters():
    param.requires_grad = True
    
for param in model.unet.zero_convs.parameters():
    param.requires_grad = True

for name,param in model.decode_head.named_parameters():
    param.requires_grad = True

trainable = [name for name, p in model.named_parameters() if p.requires_grad]
print(f"[🧠 Trainable Params]: {trainable}")


checkpoint = torch.load("/work3/s203557/checkpoints/vpd.chkpt", map_location=DEVICE)
model.load_state_dict(checkpoint, strict=False)  # allow partial load if needed
print("✅ Loaded pretrained VPD weights")

model.to(DEVICE)
model.train()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)


# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    epoch_loss = 0.0

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
                flip=False,
                flip_direction='horizontal'
            )
            for _ in range(imgs.shape[0])
        ]

        loss_dict = model.forward_train(imgs, img_metas=img_metas,
                                        gt_semantic_seg=gt_segs, gt_bboxes=batch_boxes)
        
        loss = sum(loss_dict.values())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

        # # 🧠 Visualization: Only once per epoch, on the first batch
        # if batch_idx == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         # Assume segmentation prediction via `simple_test`
        #         pred_logits = model.simple_test(imgs, img_metas, rescale=True)
        #         # Extract first image's prediction
        #         pred = pred_logits[0]  # Shape [H, W]
        #         pred_tensor = torch.tensor(pred, device=DEVICE).unsqueeze(0)  # [1, H, W]

        #         # Visualize first sample in the batch
        #         visualize_prediction(
        #             image=imgs[0].cpu(),
        #             box_map=batch_boxes[0].cpu(),
        #             pred_mask=pred_tensor.cpu(),
        #             gt_mask=gt_segs[0].cpu(),
        #             step=epoch + 1
        #         )
        #     model.train()

    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")
