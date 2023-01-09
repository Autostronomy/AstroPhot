import torch

ap_dtype = torch.float64
ap_device = "cuda:0" if torch.cuda.is_available() else "cpu"
