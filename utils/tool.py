import torch
import torch.nn.functional as F

def morphological_processing(tensor, kernel_size=3, iterations=2):
    device = tensor.device
    C, H, W = tensor.shape
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)

    input_tensor = tensor.unsqueeze(1).float()
    dilated = input_tensor

    for _ in range(iterations):
        dilated = F.conv2d(dilated, kernel, padding=kernel_size // 2)
        dilated = (dilated > 0).float()

    result = dilated.squeeze(1)
    return result
