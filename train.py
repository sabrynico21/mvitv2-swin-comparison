from sys import stderr
import torch
import torch.nn.functional as F
import constants
from torch import GradScaler, nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from functools import reduce
from charades import CharadesDataset
from parse import parse_cmd
from utils import ClipSampler, collate_fn

model, collation_method = parse_cmd()

dataset = CharadesDataset(transform=constants.PREPROCESSING_TRANSFORMS)

data_loader = DataLoader(dataset, batch_size=constants.BATCH_SIZE, collate_fn=lambda batch: collate_fn(dataset, batch, collation_method))

model.head = nn.Linear(model.head[1].in_features, constants.NUM_CLASSES).cuda()  # Adjust the final layer to the number of classes in Charades

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=constants.LEARNING_RATE)
scaler = GradScaler()
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5)

# Training loop
for epoch in range(constants.NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(data_loader):
        for video in batch:
            # print(f'''Video frames: {len(video.video)}, Frame rate: {video.framerate} Actions: {video.actions}, 
            #     Objects: {video.objects}, Timings: {video.timings}''')
            
            # for each video in input, sample n clips of 16 frames with a stride of tau (depends on framerate)
            if not video.framerate:
                print("Framerate not found in video metadata for current video. Skipping...", file=stderr)
                continue

            clip_sampler = ClipSampler(video, max_samples=12)
            samples = clip_sampler(probability=0.85)
            if not samples:
                print("Framerate not found in video metadata for current video. Skipping...", file=stderr)
                continue
            
            clips, actions, intervals = zip(*samples)
            clips = torch.stack(clips).permute((0,2,1,3,4)).cuda()

            # multiple actions may be returned for the same clip
            truth = []
            action_ids_per_clip, _ = zip(*actions)

            for clip_actions in action_ids_per_clip:
                clip_actions = [
                    F.one_hot(torch.tensor(action), num_classes=constants.NUM_CLASSES) 
                    for action in clip_actions
                ]

                if not clip_actions:
                    truth.append(constants.NO_ACTION_TENSOR)
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

        if (i+1) % constants.GRADIENT_ACCUMULATION_ITERS == 0:
            scaler.step(optimizer)
            scaler.update()
            # Zero the parameter gradients
            optimizer.zero_grad()
        # optimizer.zero_grad()
        
        average_loss = running_loss / (i+1)
        print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {average_loss:.4f}")

print("Finished Training")