import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from segmentation.dataloader.voc_dataloader import VOCDatasetWithBBoxes

from PIL import Image
from torchvision.transforms.functional import to_pil_image

def draw_boxes_to_mask(bboxes, size):
    """
    Creates a binary mask with white rectangles for each bounding box.
    - bboxes: Tensor of shape [N, 4], with each box as [x1, y1, x2, y2]
    - size: (H, W) of the output mask
    """
    mask = torch.zeros(size, dtype=torch.uint8)
    for box in bboxes:
        x1, y1, x2, y2 = box.int()
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, size[1] - 1)
        y2 = min(y2, size[0] - 1)
        mask[y1:y2, x1:x2] = 1
    return mask.unsqueeze(0)  # shape: [1, H, W]
# --- Settings ---
root_dir = "/work3/s203557/data/VOCdevkit/VOC2012"
img_size = (256, 256)
num_samples = 5
save_dir = "plots/box_gt_visualization"
os.makedirs(save_dir, exist_ok=True)

# --- Dataset ---
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])
dataset = VOCDatasetWithBBoxes(root_dir=root_dir, split="train", transform=transform)

# --- Visualize ---
for idx in range(num_samples):
    image, bboxes, seg_mask = dataset[idx]

    # Generate binary box mask
    box_mask = draw_boxes_to_mask(bboxes, size=seg_mask.shape)  # [1, H, W]

    # Visualize
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image.permute(1, 2, 0).numpy())
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(box_mask.squeeze(0).numpy(), cmap="Greens")
    axs[1].set_title("Bounding Boxes")
    axs[1].axis("off")

    axs[2].imshow(seg_mask.numpy(), cmap="tab20")
    axs[2].set_title("Ground Truth Segmentation")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"plots/fixed_sample_{idx}.png")
    plt.close()

print(f"[✅] Saved visualizations to {save_dir}")
