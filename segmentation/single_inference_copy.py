import os
import sys
import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt

# Add your module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models  # custom VPD model should be registered here
import vpd     # required for TextAdapter, UNetWrapper, etc.

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Paths ===
config_file = './configs/fpn_vpd_sd1-5_512x512_gpu8x2.py'
checkpoint_file = '/dtu/blackhole/0e/154958/checkpoints/vpd.chkpt'
#img_path = '/dtu/blackhole/0e/154958/data/ade/ADEChallengeData2016/images/training/ADE_train_00001835.jpg'
# VOC datapath: 
img_path = '/work3/s203557/data/VOCdevkit/VOC2012/JPEGImages'
# === Load model manually ===
cfg = Config.fromfile(config_file)
cfg.model.pretrained = None  # prevent loading pretrained weights accidentally
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

# Load weights without assuming 'meta'
_ = load_checkpoint(model, checkpoint_file, map_location='cpu')

# Manually set class names and palette if missing
if not hasattr(model, 'CLASSES'):
    model.CLASSES = [str(i) for i in range(150)]  # ADE20K default = 150 classes
if not hasattr(model, 'PALETTE'):
    model.PALETTE = [[i, i, i] for i in range(150)]  # grayscale fallback

# Attach config and move to device
model.cfg = cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# === Prepare single image ===
img = mmcv.imread(img_path)
test_pipeline = Compose(cfg.data.test.pipeline)
data = dict(img_info=dict(filename=img_path), img_prefix=None)
data = test_pipeline(data)
data = collate([data], samples_per_gpu=1)
data = scatter(data, [device])[0]

# === Run inference ===
with torch.no_grad():
    result = model.inference(data['img'][0], data['img_metas'][0])

# === Visualize segmentation ===
seg_pred = result[0]  # shape: (H, W), numpy array
palette = model.PALETTE

color_mask = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
for label, color in enumerate(palette):
    color_mask[seg_pred == label] = color

# === Show original + segmentation side by side ===
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

overlay = mmcv.bgr2rgb(img).copy()
alpha = 0.9 
overlay = (overlay * (1 - alpha) + color_mask * alpha).astype(np.uint8)

plt.figure(figsize=(7, 6))
plt.imshow(overlay)
plt.axis('off')
plt.title('Overlay (Image + Segmentation)')
plt.tight_layout()
plt.savefig("test_overlay.png")
plt.show()
