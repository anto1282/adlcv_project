import torch
from torchvision.utils import save_image
from vpd.models import FrozenCLIPEmbedder  # or paste the class in this file

# Pascal VOC 2012 class names
voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor"
]

# Optional: prepend "a photo of a ..." for better CLIP compatibility
voc_prompts = [f"a photo of a {cls}" for cls in voc_classes]

# Instantiate the embedder
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = FrozenCLIPEmbedder(device=device, pool=False).to(device)  # pool=False = get full sequence

# Encode prompts
with torch.no_grad():
    embeddings = embedder.encode(voc_prompts)  # shape: (20, 77, 768) for ViT-L/14

# Save or use them as needed
torch.save(embeddings, "voc2012_class_embeddings.pt")
print(f"Saved Pascal VOC embeddings: {embeddings.shape}")  # should be (20, 77, 768)