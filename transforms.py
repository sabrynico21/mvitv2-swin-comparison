import torch
import numpy as np  
import cv2
from torchvision import transforms

class ResizeVideoTransform:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, frames):
        return np.array([cv2.resize(f, self.size) for f in frames])

class VideoToTensorTransform:
    def __call__(self, frames):
        to_tensor = transforms.ToTensor()
        return torch.from_numpy(np.array([to_tensor(f) for f in frames]))
