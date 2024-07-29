import cv2
import numpy as np
import torch
from torchvision import transforms

resize_frame = lambda fs, sz: cv2.resize(fs, sz)
def timings_to_frames(ts, frame_rate):
    output = []
    for t in ts:
        output.append([
            torch.round(t[0] * frame_rate), 
            torch.round(t[1] * frame_rate)
        ])
    return output

def collate_fn(dataset, batch, mode):
    if mode == 'padding':
        return collate_padding(dataset, batch)
    else:
        return collate_trimming(dataset, batch)


def collate_padding(dataset, batch): # not working as intended
    # Find the maximum number of frames in the batch
    samples = [dataset.extract_sample(data) for data in batch]
    print(batch[0])
    max_frames = max([video['video'].shape[0] for video in samples])

    # Pad all videos to the same length
    for video in samples:
        video_frames = video['video']
        num_frames = video_frames.shape[0]
        if num_frames < max_frames:
            padding = np.zeros((max_frames - num_frames, *video_frames.shape[1:]), dtype=video_frames.dtype)
            video['video'] = np.concatenate((video_frames, padding), axis=0)
    
    # Stack the videos and actions
    videos = torch.stack([video['video'] for video in samples])
    framerates = [video['framerate'] for video in samples]
    actions = [video['actions'] for video in samples]
    objects = [video['objects'] for video in samples]
    timings = [video['timings'] for video in samples]
    
    return {
        'video': videos, 
        'framerate': framerates,
        'actions': actions, 
        'objects': objects,
        'timings': timings
    }

def collate_trimming(dataset, batch):
    # Find the minimum number of frames in the batch
    samples = [dataset.extract_sample(data) for data in batch]

    min_frames = min([video['video'].shape[0] for video in samples])

    # Prepare lists for padded videos and adjusted annotations
    cropped_videos = []
    adjusted_batch_timings = []
    adjusted_batch_actions = []

    for video in samples:
        video_frames = video['video']
        # Crop video to minimum length
        cropped_video = video_frames[:min_frames]
        cropped_videos.append(cropped_video)
        
        # Adjust annotations
        adjusted_video_timings = []
        adjusted_video_actions = []

        print(video['timings'])
        for a, t in enumerate(video['timings']):
            if t[0] < min_frames:
                # Clip end_frame if it goes beyond the cropped length
                adjusted_video_timings.append(np.clip(t, 0, min_frames))
                adjusted_video_actions.append(video['actions'][a])
        
        adjusted_batch_timings.append(torch.from_numpy(np.array(adjusted_video_timings, dtype=np.int32)))
        adjusted_batch_actions.append(np.array(adjusted_video_actions, dtype=np.int32))
    
    # Stack the cropped videos
    videos = torch.stack(cropped_videos)
    framerates = [video['framerate'] for video in samples]
    objects = [video['objects'] for video in samples]

    return {
        'video': videos, 
        'framerate': framerates,
        'actions': adjusted_batch_actions, 
        'objects': objects,
        'timings': adjusted_batch_timings
    }

class ResizeVideoTransform:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, frames):
        return np.array([resize_frame(f, self.size) for f in frames])

class VideoToTensorTransform:
    def __call__(self, frames):
        to_tensor = transforms.ToTensor()
        return torch.from_numpy(np.array([to_tensor(f) for f in frames]))
