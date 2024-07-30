import torch
import torch.nn.functional as F
from torch import GradScaler, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from argparse import ArgumentParser
from functools import reduce
from charades import CharadesDataset
from utils import ClipSampler, ResizeVideoTransform, VideoToTensorTransform, collate_fn

# /----/ CONSTANTS /----/ #
PREPROCESSING_TRANSFORMS = transforms.Compose([
    ResizeVideoTransform((224, 224)),
    VideoToTensorTransform()
])
NUM_EPOCHS = 10 
NUM_CLASSES = 158
NO_ACTION_TENSOR = F.one_hot(torch.tensor(NUM_CLASSES-1), num_classes=NUM_CLASSES)
BATCH_SIZE = 4
LEARNING_RATE = 0.001
GRADIENT_ACCUMULATION_ITERS = BATCH_SIZE

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



dataset = CharadesDataset(transform=PREPROCESSING_TRANSFORMS)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_fn(dataset, batch, args.collation))

model.head = nn.Linear(model.head[1].in_features, NUM_CLASSES).cuda()  # Adjust the final layer to the number of classes in Charades

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5)
# Training loop

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(data_loader):
        for video in batch:
            # print(f'''Video frames: {len(video.video)}, Frame rate: {video.framerate} Actions: {video.actions}, 
            #     Objects: {video.objects}, Timings: {video.timings}''')
            
            # for each video in input, sample n clips of 16 frames with a stride of tau (depends on framerate)
            clip_sampler = ClipSampler(video, max_samples=12)
            samples = clip_sampler(probability=0.85)

            clips, actions, intervals = zip(*samples)
            clips = torch.stack(clips).permute((0,2,1,3,4)).cuda()

            # multiple actions may be returned for the same clip
            truth = []
            action_ids_per_clip, _ = zip(*actions)

            for clip_actions in action_ids_per_clip:
                clip_actions = [
                    F.one_hot(torch.tensor(action), num_classes=NUM_CLASSES) 
                    for action in clip_actions
                ]

                if not clip_actions:
                    truth.append(NO_ACTION_TENSOR)
                else:
                    truth.append(reduce(torch.add, clip_actions))
            
            with torch.amp.autocast('cuda'):
                # Forward pass
                predicted = model(clips).cuda()

                # labels need to be adjusted accordingly: must be a one-hot encoded vector of 158 elements
                # consider multiple labels in a clip according to timings? (unsure)
                # compute loss on each batch of clips
                truth = torch.stack(truth).type(torch.float32).cuda()
                loss = criterion(predicted, truth)
            
            # loss.backward()
            scaler.scale(loss).backward()

        running_loss += loss.item()

        if (i+1) % GRADIENT_ACCUMULATION_ITERS == 0:
            scaler.step(optimizer)
            scaler.update()
            # Zero the parameter gradients
            optimizer.zero_grad()
        # optimizer.zero_grad()
        
        average_loss = running_loss / (i+1)
        print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {average_loss:.4f}")

print("Finished Training")