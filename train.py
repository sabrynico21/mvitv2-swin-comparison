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

model, collation_method, model_name = parse_cmd()

dataset = CharadesDataset(transform=constants.PREPROCESSING_TRANSFORMS)

data_loader = DataLoader(dataset, batch_size=constants.BATCH_SIZE, collate_fn=lambda batch: collate_fn(dataset, batch, collation_method))

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=constants.LEARNING_RATE)
scaler = GradScaler()
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5)

# Training loop
for epoch in range(constants.NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for i, batch in enumerate(data_loader):
        num_batches += 1
        batch_loss = 0.0
        num_videos = 0

        batch_truths = []
        batch_clips = []
        for video in batch:
            # for each video in input, sample n clips of x frames with a stride of tau (depends on framerate)

            try:
                clip_sampler = ClipSampler(video, clip_size=constants.SAMPLE_CLIP_SIZE)
                samples = clip_sampler()

                clips, actions, intervals = zip(*samples)
                batch_clips.extend(clips)

                # multiple actions may be returned for the same clip
                action_ids_per_clip, _ = zip(*actions)

                for clip_actions in action_ids_per_clip:
                    clip_actions = [
                        F.one_hot(torch.tensor(action), num_classes=constants.NUM_CLASSES) 
                        for action in clip_actions
                    ]

                    if not clip_actions:
                        batch_truths.append(constants.NO_ACTION_TENSOR)
                    else:
                        batch_truths.append(reduce(torch.add, clip_actions))
            except:
                print("Couldn't extract any sample for current video. Skipping...", file=stderr)
                continue
        # print(len(batch_clips), file=stderr)
        clips = torch.stack(batch_clips).permute((0,2,1,3,4)).cuda()
        with torch.amp.autocast('cuda'):
            # Forward pass
            predicted = model(clips).cuda()

            # labels need to be adjusted accordingly: must be a one-hot encoded vector of 158 elements
            # consider multiple labels in a clip according to timings? (unsure)
            # compute loss on each batch of clips
            truth = torch.stack(batch_truths).type(torch.float32).cuda()
            loss = criterion(predicted, truth)
            batch_loss += loss.item()
            num_videos += 1
            
        scaler.scale(loss).backward()

        batch_loss /= num_videos

        epoch_loss += batch_loss

        if (i+1) % constants.GRADIENT_ACCUMULATION_ITERS == 0:
            scaler.step(optimizer)
            scaler.update()
            # Zero the parameter gradients
            optimizer.zero_grad()
        
        if (i+1) % constants.PRINT_BATCH_LOSS_EVERY == 0:
            print(f"Epoch [{epoch + 1}/{constants.NUM_EPOCHS}], Batch [{i+1}], Loss: {batch_loss:.4f}", file=stderr)
            
    average_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch + 1}/{constants.NUM_EPOCHS}], Average Loss: {average_loss:.4f}", file=stderr)
    torch.save(model, f'{model_name}_epoch_{epoch+1}.pth')
    scheduler.step()

print("Finished Training")