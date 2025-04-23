import torch

# Load your checkpoint
checkpoint_path = '/work3/s203557/checkpoints/vpd_1.chkpt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Print keys to see what's inside
print("Checkpoint keys:")
print(checkpoint.keys())

# Usually PyTorch checkpoints have either:
# - 'state_dict' (common in mmseg, mmcv, stable diffusion, etc.)
# - or just model parameters directly stored in the root.

# Check if 'state_dict' is available:
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("\nState_dict keys (first 20 shown):")
    for key in list(state_dict.keys())[:20]:
        print(key)
else:
    print("\nCheckpoint parameters (first 20 shown):")
    for key in list(checkpoint.keys())[:20]:
        print(key)

# Specifically check encoder_vq parameters:
vq_params = {k: v for k, v in checkpoint.get('state_dict', checkpoint).items() if 'encoder_vq' in k}
if vq_params:
    print("\nEncoder VQ parameters found:")
    for key in vq_params:
        print(key)
else:
    print("\nNo Encoder VQ parameters found in the checkpoint.")


