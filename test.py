import torch
import numpy as np
import constants
import torch.nn.functional as F
from torcheval.metrics import TopKMultilabelAccuracy, MultilabelAccuracy
from sys import stderr
from utils import ClipSampler, collate_fn, load_action_weights
from charades import CharadesDataset
from torch.utils.data import DataLoader
from torch import nn
from functools import reduce
from parse import parse_cmd

model, collation_method, _ = parse_cmd()

dataset = CharadesDataset(transform=constants.PREPROCESSING_TRANSFORMS, split="test")
test_loader = DataLoader(dataset, batch_size=constants.BATCH_SIZE, collate_fn=lambda batch: collate_fn(dataset, batch, collation_method))

actions_weights = load_action_weights()

criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=actions_weights.cuda())
model.eval()

all_predictions = []
all_truths = []
total_loss = 0.0
num_batches = 0

with torch.no_grad():
    for batch in test_loader:
        #batch_loss = 0.0
        #num_videos = 0

        batch_truths = []
        batch_clips = []
        for video in batch:
                # for each video in input, sample n clips of x frames with a stride of tau (depends on framerate)
            try:    
                #clip_sampler = ClipSampler(video, clip_size=constants.SAMPLE_CLIP_SIZE, max_samples=5, mode='contiguous')
                clip_sampler = ClipSampler(video, clip_size=constants.SAMPLE_CLIP_SIZE)
                samples = clip_sampler(probability=0.50)
          
                clips, actions, intervals = zip(*samples)
                batch_clips.extend(clips)

                # multiple actions may be returned for the same clip
                action_ids_per_clip, _ = zip(*actions)
                
                for clip_actions in action_ids_per_clip:
                    if not clip_actions:
                        print("No actions found in current clip. Skipping...", file=stderr)
                        continue

                    clip_actions = [
                        F.one_hot(torch.tensor(action), num_classes=constants.NUM_CLASSES) 
                        for action in clip_actions
                    ]
                    batch_truths.append(reduce(torch.add, clip_actions))
            except:
                print("Couldn't extract any sample from current video. Skipping...", file=stderr)
                continue
                
        if not batch_clips:
            print("Couldn't extract any clip from current batch. Skipping...", file=stderr)
            continue

        clips = torch.stack(batch_clips).permute((0,2,1,3,4)).cuda()
        with torch.amp.autocast('cuda'):
        
            predicted = model(clips).cuda()
            truth = torch.stack(batch_truths).type(torch.float32).cuda()
            loss = criterion(predicted, truth)
            total_loss += loss.item()
            #num_videos += 1

            predictions = torch.sigmoid(predicted).cpu().numpy()  # Apply sigmoid to get probabilities
            truth = truth.cpu().numpy()
            
            all_predictions.append(predictions)
            all_truths.append(truth)
        
        #batch_loss /= num_videos
        #total_loss += batch_loss
        num_batches += 1

        print(f"Batch: {num_batches}, Loss: {loss.item():.4f}", file=stderr)
        
average_loss = total_loss / num_batches

print(f'Average Loss: {average_loss:.4f}', file=stderr)

all_predictions = np.vstack(all_predictions)
all_truths = np.vstack(all_truths)
all_predictions_tensor = torch.tensor(all_predictions, dtype=torch.float32)
all_truths_tensor = torch.tensor(all_truths, dtype=torch.float32)

for mode in ["exact_match","hamming","overlap","contain","belong"]:
    metric = MultilabelAccuracy(criteria=mode)
    metric.update(all_predictions_tensor, all_truths_tensor)
    accuracy = metric.compute()
    print(f"Mode: {mode}, Accuracy: {accuracy.item()}", file=stderr)

for mode in ["exact_match","hamming","overlap","contain","belong"]:
    for k in [2,3,5,10]:
        top_k_accuracy = TopKMultilabelAccuracy(k = k, criteria=mode)
        top_k_accuracy.update(all_predictions_tensor, all_truths_tensor)
        accuracy = top_k_accuracy.compute()
        print(f"Mode: {mode}, Top-{k} Accuracy: {accuracy.item()}", file=stderr)

