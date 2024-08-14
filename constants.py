
import torch
import torch.nn.functional as F
from torchvision import transforms
from utils import ResizeVideoTransform, VideoToTensorTransform

PREPROCESSING_TRANSFORMS = transforms.Compose([
    ResizeVideoTransform((224, 224)),
    VideoToTensorTransform()
])
NUM_EPOCHS = 3
NUM_CLASSES = 158
NO_ACTION_TENSOR = F.one_hot(torch.tensor(NUM_CLASSES-1), num_classes=NUM_CLASSES)
BATCH_SIZE = 18
LEARNING_RATE = 0.001
GRADIENT_ACCUMULATION_ITERS = 4
PRINT_BATCH_LOSS_EVERY = 4
VIDEO_MAX_SAMPLES = 1
SAMPLE_CLIP_SIZE = 16
