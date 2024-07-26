import cv2
import numpy as np
import torch

resize_frame = lambda fs, sz: cv2.resize(fs, sz)
def timings_to_frames(ts, frame_rate):
    output = []
    for t in ts:
        output.append([
            torch.round(t[0] * frame_rate), 
            torch.round(t[1] * frame_rate)
        ])
    return output


def collate_padding(batch):
    # Find the maximum number of frames in the batch
    print(batch[0])
    max_frames = max([video['video'].shape[0] for video in batch])

    # Pad all videos to the same length
    for video in batch:
        video_frames = video['video']
        num_frames = video_frames.shape[0]
        if num_frames < max_frames:
            padding = np.zeros((max_frames - num_frames, *video_frames.shape[1:]), dtype=video_frames.dtype)
            video['video'] = np.concatenate((video_frames, padding), axis=0)
    
    # Stack the videos and actions
    videos = torch.stack([torch.from_numpy(video['video']) for video in batch])
    actions = [video['actions'] for video in batch]
    objects = [video['objects'] for video in batch]
    timings = [video['timings'] for video in batch]
    
    return {
        'video': videos, 
        'actions': actions, 
        'objects': objects,
        'timings': timings
    }

def collate_trimming(batch):
    # Find the minimum number of frames in the batch
    min_frames = min([video['video'].shape[0] for video in batch])

    # Prepare lists for padded videos and adjusted annotations
    cropped_videos = []
    adjusted_batch_timings = []
    adjusted_batch_actions = []

    for video in batch:
        video_frames = video['video']
        
        # Crop video to minimum length
        cropped_video = video_frames[:min_frames]
        cropped_videos.append(torch.from_numpy(cropped_video))
        
        # Adjust annotations
        adjusted_video_timings = []
        adjusted_video_actions = []
        for a, t in enumerate(video['timings']):
            tframes = timings_to_frames(t, video['framerate'])
            
            if tframes[0] < min_frames:
                # Clip end_frame if it goes beyond the cropped length
                if tframes[1] > min_frames:
                    tframes[1] = min_frames
                adjusted_video_timings.append(tframes)
                adjusted_video_actions.append(video['actions'][a])
        
        adjusted_batch_timings.append(adjusted_video_timings)
        adjusted_batch_actions.append(adjusted_video_actions)
    
    # Stack the cropped videos
    videos = torch.stack(cropped_videos)
    actions = torch.stack(adjusted_batch_actions)
    objects = [video['objects'] for video in batch]
    timings = torch.stack(adjusted_batch_timings)

    return {
        'video': videos, 
        'actions': actions, 
        'objects': objects,
        'timings': timings
    }

class ResizeVideoTransform:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, frames):
        return np.array([resize_frame(f, self.size) for f in frames])

# class VideoToTensorTransform:
#     def __call__(self, frames):
#         to_tensor = transforms.ToTensor()
#         return torch.from_numpy(np.array([to_tensor(f) for f in frames]))
