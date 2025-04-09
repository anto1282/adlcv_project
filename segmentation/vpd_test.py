import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom modules
from segmentation.dataloader.voc_dataloader import VOCDataset
from segmentation.models.vpd_seg import VPDSeg

# --- Configs ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running on device:", DEVICE)
IMG_SIZE = (64, 64)  # Reduce from (128, 128) to (64, 64)

# --- Dataset & Dataloader ---
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

test_dataset = VOCDataset(root_dir='/work3/s203557/data/VOCdevkit/VOC2012', split='val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- Model ---
model = VPDSeg(
    base_size=IMG_SIZE[0],
    decode_head=dict(
        type='FCNHead',
        in_channels=1280,
        channels=128,
        num_convs=2,
        kernel_size=3,
        num_classes=21,  # Pascal VOC has 21 classes
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=False),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    sd_path='/work3/s203557/checkpoints/v1-5-pruned-emaonly.ckpt',
    sd_config='v1-inference.yaml',
    class_embedding_path='class_embeddings.pth',
    test_cfg=dict(mode='whole')  # Ensure test_cfg is properly defined
)

# Clear GPU memory before loading the model
if DEVICE == 'cuda':
    torch.cuda.empty_cache()

# Load checkpoint on CPU first to avoid GPU memory issues
checkpoint = torch.load("/work3/s203557/checkpoints/vpd.chkpt", map_location='cpu')
model.load_state_dict(checkpoint, strict=False)

# Try to move the model to the GPU
try:
    model.to(DEVICE)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("⚠️ CUDA out of memory. Switching to CPU.")
        DEVICE = 'cpu'
        model.to(DEVICE)
    else:
        raise e

# Enable mixed precision to reduce memory usage (optional)
scaler = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None

model.eval()
print("✅ Model loaded and ready for inference.")

# --- IoU Metric ---
def compute_iou(pred, target, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            iou_per_class.append(float('nan'))  # Ignore this class
        else:
            iou_per_class.append(intersection / union)
    return np.nanmean(iou_per_class)  # Mean IoU ignoring NaNs

# --- Inference and Evaluation ---
iou_scores = []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        # Directly move images and labels to the device
        images = images.to(DEVICE)
        labels = labels.squeeze(1).to(DEVICE)  # Remove channel dimension if present

        # Create img_metas (metadata about the input images)
        img_metas = [{'img_shape': images.shape[2:], 'ori_shape': images.shape[2:], 'pad_shape': images.shape[2:]}]

        # Forward pass using simple_test for inference with mixed precision
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model.simple_test(images, img_metas, rescale=True)
        preds = torch.argmax(torch.tensor(outputs[0]), dim=0)  # Convert output to tensor and get predictions

        # Compute IoU
        iou = compute_iou(preds.cpu(), labels.cpu(), num_classes=21)
        iou_scores.append(iou)

# Report Mean IoU
mean_iou = np.nanmean(iou_scores)
print(f"Mean IoU: {mean_iou:.4f}")
