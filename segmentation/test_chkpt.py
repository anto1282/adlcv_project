import torch
ckpt = torch.load('/work3/s203557/experiments/control_net_vpd/iter_100.pth', map_location='cpu')
print([k for k in ckpt['state_dict'].keys() if 'zero_conv' in k])
