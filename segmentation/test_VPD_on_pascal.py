import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import masks_to_boxes

import mmcv
from mmcv.runner import load_checkpoint
from mmcv import ConfigDict
from mmseg.models import build_segmentor
from mmseg.models.builder import SEGMENTORS
from tqdm import tqdm
import matplotlib.pyplot as plt


# -------------------------------
# Add the parent directory to the Python path to allow imports from sibling modules
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom segmentation model and dataset
from segmentation.models.vpd_seg_og import VPDSeg
from segmentation.dataloader.voc_dataloader import VOCDatasetWithBBoxes
import torch.nn as nn

# -------------------------------
# Helper function: Evaluate segmentation performance
# -------------------------------
def evaluate_segmentation(preds, gts, num_classes=21):
    """
    Evaluate segmentation performance using metrics such as aAcc, mAcc, and mIoU.

    Args:
        preds (list): List of predicted masks.
        gts (list): List of ground truth masks.
        num_classes (int): Number of classes in the dataset.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    def fast_hist(a, b, n):
        # Compute a histogram for evaluating IoU
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k].astype(int),
                           minlength=n**2).reshape(n, n)
    
    hist = np.zeros((num_classes, num_classes))
    for pred, gt in zip(preds, gts):
        pred = pred.cpu().numpy().flatten()
        gt = gt.cpu().numpy().flatten()
        hist += fast_hist(gt, pred, num_classes)
    
    # Compute overall accuracy, mean accuracy, and mean IoU
    aAcc = np.diag(hist).sum() / hist.sum()
    class_acc = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
    mAcc = np.nanmean(class_acc)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
    mIoU = np.nanmean(iou)
    return {"aAcc": aAcc, "mAcc": mAcc, "mIoU": mIoU}

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

# -------------------------------
# Model configuration and build
# -------------------------------
checkpoint_path = "/work3/s203557/checkpoints/vpd.chkpt"  # Path to the model checkpoint
DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
print("Training on device",DEVICE)
BATCH_SIZE = 2
NUM_EPOCHS = 1
LR = 1e-4

# --- Dataset & Dataloader ---
IMG_SIZE = (128, 128)  # or 256x256, 384x384, etc.

# Load the model checkpoint
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Define the model configuration
model = VPDSeg(base_size = IMG_SIZE[0],
    decode_head=dict(
        type='FCNHead',
        in_channels=1280,  # adjust based on your unet wrapper output
        channels=128,
        num_convs=2,
        kernel_size=3,
        num_classes=21,  # for VOC
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=False),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
    ),
    sd_path='/work3/s203557/checkpoints/v1-5-pruned-emaonly.ckpt',
    sd_config='v1-inference.yaml',
    class_embedding_path='class_embeddings.pth',
    test_cfg=ConfigDict(dict(mode='whole'))

)

# Load the checkpoint into the model
checkpoint = torch.load("/work3/s203557/checkpoints/vpd.chkpt", map_location=DEVICE)
model.load_state_dict(checkpoint, strict=False)

model.to(DEVICE)  # Move the model to the specified device
model.eval()  # Set the model to evaluation mode

# Set the model's class names and palette if missing
if not hasattr(model, 'CLASSES'):
    model.CLASSES = [str(i) for i in range(21)]  # VOC has 21 classes including background
if not hasattr(model, 'PALETTE'):
    model.PALETTE = [[i, i, i] for i in range(21)]  # grayscale fallback

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Resize images to the model's input size
    transforms.ToTensor()  # Convert images to tensors
])

# Create the test dataset
test_dataset = VOCDatasetWithBBoxes(
    root_dir='/work3/s203557/data/VOCdevkit/VOC2012',  # Path to the dataset
    split='val',  # Use the validation split
    transform=transform  # Apply transformations
)

# Create the DataLoader for the test dataset
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,  # Batch size of 1 for inference
    collate_fn=lambda x: tuple(zip(*x))  # Custom collate function
)


pbar = tqdm(test_loader, desc="Inference", total=len(test_loader))

for batch_idx, (imgs, boxes, segs) in enumerate(pbar):
    try:
        imgs = torch.stack(imgs).to(DEVICE)  
        H, W = imgs.shape[2], imgs.shape[3]

        gt_segs = torch.stack([torch.tensor(seg, dtype=torch.long) for seg in segs]).to(DEVICE)
        gt_segs = gt_segs.unsqueeze(1)

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

        # Perform inference
        with torch.no_grad():
            pred_logits = model.simple_test(imgs, img_metas, rescale=True)
            # Extract first image's prediction
            pred = pred_logits[0]
            pred_tensor = torch.tensor(pred, device=DEVICE).unsqueeze(0)  # [1, H, W]
            pred_np = pred_tensor.cpu().numpy().squeeze(0)  # [H, W]
            gt_np = gt_segs[0].cpu().numpy().squeeze(0)  # [H, W]

            # evaluate segmentation performance
            # metrics = evaluate_segmentation([pred_tensor], [gt_segs[0]], num_classes=21)
            # pbar.set_postfix(**metrics)
            # Visualize first sample in the batch
            visualize_prediction(
                image=imgs[0].cpu(),
                box_map=boxes[0].cpu(),
                pred_mask=pred_tensor.cpu(),
                gt_mask=gt_segs[0].cpu(),
                step=batch_idx
            )

    except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory at batch {batch_idx}. Skipping this batch.")
                torch.cuda.empty_cache()  # Clear the cache to free up memory
            else:
                raise e

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
    #             pred_mask=pred_tensor.cpu(),
    #             gt_mask=gt_segs[0].cpu(),
    #         )

        
    


# -------------------------------
# Main function: DataLoader creation, inference, and evaluation
# -------------------------------
# def main():
#     """
#     Main function to load the dataset, perform inference, and evaluate the model.
#     """

#     img_metas = [
#             dict(
#                 ori_shape=(int(H), int(W)),
#                 img_shape=(int(H), int(W)),
#                 scale_factor=(1.0, 1.0),
#                 flip=False,
#                 flip_direction='horizontal'
#             )
#             for _ in range(imgs.shape[0])
#         ]
    
#     print(f"Loaded {len(test_dataset)} samples for testing.")
#     print("🚀 Running custom inference loop...")
    
#     # Perform inference
#     seg_model.eval()  # Set the model to evaluation mode
#     seg_model.to(DEVICE)  # Move the model to the specified device
#     output = seg_model.inference(test_loader, , rescale=False)
#     #preds, gts = custom_test(seg_model, test_loader, DEVICE)
    
#     print("\n📊 Evaluating performance (aAcc, mAcc, mIoU)...")
    
#     # Evaluate the model's performance
#     metrics = evaluate_segmentation(preds, gts, num_classes=21)
    
#     print("\n✅ Evaluation Results:")
#     for metric_name, metric_value in metrics.items():
#         print(f"{metric_name}: {metric_value:.4f}")

# # Entry point of the script
# if __name__ == '__main__':
#     main()

    