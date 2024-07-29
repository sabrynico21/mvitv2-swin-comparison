from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torch.optim import Adam
from charades import CharadesDataset
from utils import ResizeVideoTransform, VideoToTensorTransform, collate_fn

parser = ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    required=True,
    help="Model to train",
    choices=["mvitv2", "swin_t", "swin_s", "swin_b"],
)
parser.add_argument(
    "-C",
    "--collation",
    default='padding',
    help='Choose the collation mode: determine if videos length must be uniformed through padding or trimming'
)

args = parser.parse_args()

match args.model:
    case 'mvitv2':
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1).cuda()
    case 'swin_t':
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1).cuda()
    case 'swin_s':
        from torchvision.models.video import swin3d_s, Swin3D_S_Weights
        model = swin3d_s(weights=Swin3D_S_Weights.KINETICS400_V1).cuda()
    case 'swin_b':
        from torchvision.models.video import swin3d_b, Swin3D_B_Weights
        model = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1).cuda()
    case _:
        raise Exception


transform = transforms.Compose([
    ResizeVideoTransform((224, 224)),
    VideoToTensorTransform()
])

dataset = CharadesDataset(transform=transform)

data_loader = DataLoader(dataset, batch_size=4, collate_fn=lambda batch: collate_fn(dataset, batch, args.collation))

num_classes = 157
model.head = nn.Linear(model.head[1].in_features, num_classes)  # Adjust the final layer to the number of classes in Charades

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(data_loader):
        print(f"Video frames: {len(data["video"][0])}, Actions: {data["actions"]}")
        print(f"Objects: {data["objects"]}, Timings: {data["timings"]}")
        inputs = torch.permute(data["video"], (0,2,1,3,4)).cuda()
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        labels = torch.stack(torch.tensor(np.array(data["actions"]))).cuda()
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:    # Print every 10 batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

print("Finished Training")