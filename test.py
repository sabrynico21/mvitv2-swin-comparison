import torch
from argparse import ArgumentParser
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, hamming_loss
from utils import ClipSampler, ResizeVideoTransform, VideoToTensorTransform, collate_fn
from torchvision import transforms
from charades import CharadesDataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from functools import reduce
from parse import parse_cmd

PREPROCESSING_TRANSFORMS = transforms.Compose([
    ResizeVideoTransform((224, 224)),
    VideoToTensorTransform()
])

BATCH_SIZE = 4
NUM_CLASSES = 158
NO_ACTION_TENSOR = torch.nn.functional.one_hot(torch.tensor(NUM_CLASSES-1), num_classes=NUM_CLASSES)

model, collation_method = parse_cmd()

dataset = CharadesDataset(transform=PREPROCESSING_TRANSFORMS, split="test")
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_fn(dataset, batch, collation_method))

criterion = nn.BCEWithLogitsLoss()
model.eval()

all_predictions = []
all_truths = []
total_loss = 0.0
num_batches = 0

with torch.no_grad():
    for batch in test_loader:
        for video in batch:
            clip_sampler = ClipSampler(video, max_samples=12)
            samples = clip_sampler(probability=0.85)

            clips, actions, intervals = zip(*samples)
            clips = torch.stack(clips).permute((0, 2, 1, 3, 4)).cuda()

            truth = []
            action_ids_per_clip, _ = zip(*actions)

            for clip_actions in action_ids_per_clip:
                clip_actions = [
                    torch.nn.functional.one_hot(torch.tensor(action), num_classes=NUM_CLASSES) 
                    for action in clip_actions
                ]

                if not clip_actions:
                    truth.append(NO_ACTION_TENSOR)
                else:
                    truth.append(reduce(torch.add, clip_actions))

            truth = torch.stack(truth).type(torch.float32).cuda()

            outputs = model(clips).cuda()
            loss = criterion(outputs, truth)

            predictions = torch.sigmoid(outputs).cpu().numpy()  # Apply sigmoid to get probabilities
            truth = truth.cpu().numpy()

            all_predictions.append(predictions)
            all_truths.append(truth)

            total_loss += loss.item()
            num_batches += 1

average_loss = total_loss / num_batches
print(f'Total Loss: {total_loss:.4f}')
print(f'Average Loss: {average_loss:.4f}')

all_predictions = np.vstack(all_predictions)
all_truths = np.vstack(all_truths)

# Binarize predictions at a threshold (0.5 is commonly used)
binary_predictions = (all_predictions >= 0.5).astype(int)

# Compute metrics
precision = precision_score(all_truths, binary_predictions, average='macro')
recall = recall_score(all_truths, binary_predictions, average='macro')
f1 = f1_score(all_truths, binary_predictions, average='macro')
roc_auc = roc_auc_score(all_truths, all_predictions, average='macro')
average_precision = average_precision_score(all_truths, all_predictions, average='macro')
hamming = hamming_loss(all_truths, binary_predictions)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'Average Precision: {average_precision:.4f}')
print(f'Hamming Loss: {hamming:.4f}')


