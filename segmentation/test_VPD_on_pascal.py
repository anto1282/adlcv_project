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

# -------------------------------
# In segmentation/models/vpd_seg.py
# -------------------------------
# Ensure you use __file__ instead of _file_
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn

class VPDSeg(nn.Module):
    def __init__(self, **kwargs):
        super(VPDSeg, self).__init__()
        # Your initialization code here...
        # For demonstration, we create a dummy layer
        self.conv = nn.Conv2d(3, 21, kernel_size=1)

    def forward(self, img, img_metas, **kwargs):
        # Dummy forward: Assume each image in img is a tensor of shape [C, H, W]
        outputs = []
        for x in img:
            # Process image through a dummy convolution and expand dims to mimic [1, C, H, W]
            out = self.conv(x.unsqueeze(0))
            outputs.append(out)
        return outputs

# -------------------------------
# Register the custom model if not already registered
# -------------------------------
SEGMENTORS.register_module(module=VPDSeg)

# -------------------------------
# Custom Dataset: VOCDatasetWithBBoxes (ignoring bboxes for evaluation)
# -------------------------------
class VOCDatasetWithBBoxes(Dataset):
    def __init__(self, root_dir, split="train", transform=None, img_size=(512,512)):
        self.img_size = img_size
        self.root_dir = root_dir
        self.to_tensor = transforms.ToTensor()
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.mask_dir = os.path.join(root_dir, "SegmentationClass")
        self.transform = transform

        # Load file list from split file
        split_file = os.path.join(root_dir, "ImageSets", "Segmentation", f"{split}.txt")
        with open(split_file, "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

        # PASCAL VOC classes (background is 0)
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        # Load image and corresponding mask
        img_path = os.path.join(self.image_dir, f"{name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # color PNG mask

        # Resize mask (using NEAREST to preserve label values)
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = np.array(mask)  # shape: [H, W]
        mask = torch.from_numpy(mask).long()  # ground truth segmentation mask

        # (Optional) extract bounding boxes (ignored during evaluation)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]
        bboxes = []
        labels = []
        for obj_id in obj_ids:
            binary_mask = (mask == obj_id).to(torch.uint8)
            if binary_mask.sum() == 0:
                continue
            boxes = masks_to_boxes(binary_mask[None])[0]
            bboxes.append(boxes)
            labels.append(obj_id)
        if bboxes:
            bboxes = torch.stack(bboxes)
            labels = torch.tensor(labels)
        else:
            bboxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)
        # Return image, bounding boxes, and ground-truth segmentation mask.
        return image, bboxes, mask

# -------------------------------
# Helper function: Evaluate segmentation performance
# -------------------------------
def evaluate_segmentation(preds, gts, num_classes=21):
    def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k].astype(int),
                           minlength=n**2).reshape(n, n)
    
    hist = np.zeros((num_classes, num_classes))
    for pred, gt in zip(preds, gts):
        pred = pred.cpu().numpy().flatten()
        gt = gt.cpu().numpy().flatten()
        hist += fast_hist(gt, pred, num_classes)
    
    aAcc = np.diag(hist).sum() / hist.sum()
    class_acc = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
    mAcc = np.nanmean(class_acc)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
    mIoU = np.nanmean(iou)
    return {"aAcc": aAcc, "mAcc": mAcc, "mIoU": mIoU}

# -------------------------------
# Custom inference loop for segmentation with dummy img_metas
# -------------------------------
def custom_test(seg_model, data_loader, device):
    seg_model.eval()
    all_preds = []
    all_gts = []
    for batch in data_loader:
        images, bboxes, masks = batch
        images = [img.to(device) for img in images]
        # Create dummy metadata for each image
        img_metas = []
        for img in images:
            # Create a dummy meta dictionary. Adjust fields if your model requires more detailed info.
            meta = {
                'ori_shape': tuple(img.shape[1:]),  # (H, W)
                'img_shape': tuple(img.shape[1:]),
                'pad_shape': tuple(img.shape[1:]),
                'scale_factor': 1.0,
                'flip': False
            }
            img_metas.append(meta)

        with torch.no_grad():
            outputs = seg_model(img=images, img_metas=img_metas, return_loss=False, rescale=True)
            if not isinstance(outputs, list):
                outputs = [outputs]
            for out in outputs:
                if out.dim() == 4 and out.size(0) == 1:
                    out = out.squeeze(0)
                pred_mask = out.argmax(dim=0)
                all_preds.append(pred_mask)
        for gt in masks:
            all_gts.append(gt)
        torch.cuda.empty_cache()
    return all_preds, all_gts

# -------------------------------
# Model configuration and build
# -------------------------------
IMG_SIZE_MODEL = (512, 512)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "/work3/s203557/checkpoints/vpd.chkpt"

checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

model_cfg = dict(
    type='VPDSeg',
    base_size=IMG_SIZE_MODEL[0],
    decode_head=dict(
        type='FCNHead',
        in_channels=1280,
        channels=128,
        num_convs=2,
        kernel_size=3,
        num_classes=21,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=False),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    sd_path='/work3/s203557/checkpoints/v1-5-pruned-emaonly.ckpt',
    sd_config='v1-inference.yaml',
    class_embedding_path='class_embeddings.pth',
    test_cfg=dict(mode='whole')
)
cfg = ConfigDict(
    dict(
        model=model_cfg,
        device=DEVICE,
    )
)
cfg.model.train_cfg = None
seg_model = build_segmentor(cfg.model)
seg_model.load_state_dict(checkpoint, strict=False)
seg_model = seg_model.to(DEVICE)
seg_model.eval()

# -------------------------------
# Main function: DataLoader creation, inference, and evaluation
# -------------------------------
def main():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE_MODEL),
        transforms.ToTensor()
    ])
    test_dataset = VOCDatasetWithBBoxes(
        root_dir='/work3/s203557/data/VOCdevkit/VOC2012',
        split='val',  # change to 'test' if available
        transform=transform,
        img_size=IMG_SIZE_MODEL
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    print(f"Loaded {len(test_dataset)} samples for testing.")
    print("🚀 Running custom inference loop...")
    preds, gts = custom_test(seg_model, test_loader, DEVICE)
    print("\n📊 Evaluating performance (aAcc, mAcc, mIoU)...")
    metrics = evaluate_segmentation(preds, gts, num_classes=21)
    print("\n✅ Evaluation Results:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

if __name__ == '__main__':
    main()
