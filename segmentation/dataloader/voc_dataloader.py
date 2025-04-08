import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.ops import masks_to_boxes

class VOCDatasetWithBBoxes(Dataset):
    def __init__(self, root_dir, split="train", transform=None, img_size = (512,512)):
        self.img_size = img_size
        self.root_dir = root_dir
        self.to_tensor = transforms.ToTensor()
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.mask_dir = os.path.join(root_dir, "SegmentationClass")
        self.transform = transform

        # Load file list
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

        # Load image and mask
        img_path = os.path.join(self.image_dir, f"{name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        image = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)              # Load color PNG mask
        mask = mask.resize(self.img_size, Image.NEAREST)  # Resize safely
        mask = np.array(mask)                     # Now shape is [H, W]
        mask = torch.from_numpy(mask).long()      # Convert to torch tensor
        

        # Get unique object classes present (excluding background 0)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        bboxes = []
        labels = []

        for obj_id in obj_ids:
            # Create a binary mask for the object
            binary_mask_tensor = (mask == obj_id).to(torch.uint8)

            # Find bounding box using torchvision
            if binary_mask_tensor.sum() == 0:
                continue  # skip empty masks

            boxes = masks_to_boxes(binary_mask_tensor[None])[0]  # Single mask
            bboxes.append(boxes)
            labels.append(obj_id)



        # Stack bounding boxes
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

        return image, bboxes, mask
