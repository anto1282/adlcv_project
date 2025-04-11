import torch

# Load the file
path = 'class_embeddings.pth'
data = torch.load(path, map_location='cpu')  # use map_location='cpu' if you're not using a GPU

# Print info about the data
print("Type of loaded data:", type(data))

# If it's a tensor where each row is an embedding for a class
if isinstance(data, torch.Tensor):
    num_classes = data.shape[0]
    print("Number of classes:", num_classes)

# If it's a dictionary (e.g., {class_name: embedding})
elif isinstance(data, dict):
    num_classes = len(data)
    print("Number of classes:", num_classes)
    print("Class names:", list(data.keys()))

# If it's something else, you might need to explore its structure
else:
    print("Loaded data structure is not directly supported. Keys or structure:", data)
