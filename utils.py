import warnings
import cv2
import numpy as np
import torch
from collections import OrderedDict
from typing import List
from torchvision import transforms


resize_frame = lambda f, sz: cv2.resize(f, sz)

def collate_fn(dataset, batch, mode):
    if mode == 'padding':
        return collate_padding(dataset, batch)
    else:
        return collate_trimming(dataset, batch)



def collate_padding(dataset, batch): # not working as intended
    from charades import CharadesSample

    # Find the maximum number of frames in the batch
    samples: List[CharadesSample] = [dataset.extract_sample(data) for data in batch]
    max_frames = max([sample.video.shape[0] for sample in samples])

    # Pad all videos to the same length
    for sample in samples:
        video_frames = sample.video
        nframes = video_frames.shape[0]
        if nframes < max_frames:
            padding = np.zeros((max_frames - nframes, *video_frames.shape[1:]), dtype=video_frames.dtype)
            sample.video = np.concatenate((video_frames, padding), axis=0)

    return samples

def collate_trimming(dataset, batch):
    # Find the minimum number of frames in the batch
    from charades import CharadesSample
    samples: List[CharadesSample] = [dataset.extract_sample(data) for data in batch]

    min_frames = min([sample.video.shape[0] for sample in samples])

    # Prepare lists for padded videos and adjusted annotations
    for sample in samples:
        sample.video = sample.video[:min_frames]
        # Crop video to minimum length
        
        # Adjust annotations
        adjusted_video_timings = []
        adjusted_video_actions = []

        for a, t in enumerate(sample.timings):
            if t[0] < min_frames:
                # Clip end_frame if it goes beyond the cropped length
                adjusted_video_timings.append(np.clip(t, 0, min_frames))
                adjusted_video_actions.append(sample.actions[a])
        
        sample.timings = np.array(adjusted_video_timings, dtype=int)
        sample.actions = np.array(adjusted_video_actions, dtype=int)

    return samples


class ResizeVideoTransform:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, frames):
        return np.array([resize_frame(f, self.size) for f in frames])

class VideoToTensorTransform:
    def __call__(self, frames):
        to_tensor = transforms.ToTensor()
        return torch.from_numpy(np.array([to_tensor(f) for f in frames]))

class ClipSampler:
    def __init__(self, data, stride_range: tuple = (6,12), max_samples: int = None, mode='sequential'):
        self._data = data
        self.video = data.video
        self.framerate = data.framerate
        self.wsize = 16
        self.stride = round(min(stride_range[0] + self.framerate/2, stride_range[1] / (30.0/self.framerate)))
        self.nframes = self.video.size(0)
        self.curr = 0
        self.sample_limit = max_samples

    def __iter__(self):
        self.curr = 0
        return self
    
    def _sample(self, single_action=False):
        # take the clip
        cstart = self.curr
        cend = cstart + self.wsize
        clip = self.video[cstart:cend]
        self.curr += self.stride
        clip_actions = OrderedDict()
        for tidx, action in enumerate(self._data.actions):
            tstart, tend = self._data.timings[tidx]
            if tstart <= cstart <= tend:
                clip_actions.update({action: tend - cstart})
        if not clip_actions:
            actions = ([],0)
        else: 
            # get the most present action: 
            # for each action check the interval of action start-end timing which overlaps the clip
            # take the max
            if single_action:
                action_id = max(clip_actions, key=clip_actions.get)
                max_overlap = clip_actions[action_id]
                actions = ([action_id], max_overlap)
            # get multiple actions if their overlap with the clip is maximum
            else:
                max_overlap = max(clip_actions.values())
                action_ids = [id for id in clip_actions.keys() if clip_actions.get(id) == max_overlap]
                actions = (action_ids, max_overlap)
        
        return clip, actions, (cstart, cend)
    
    def __next__(self):
        """
        Samples overlapping clips from video frames represented as a PyTorch tensor.

        Args:
        - video_tensor (torch.Tensor): Tensor of video frames with shape (nframes, channels, height, width)
        - wsize (int): Number of frames in a clip.
        - stride (int): Number of frames to move for each new clip.

        Returns:
        - clip (torch.Tensor): Sampled clip, with shape (wsize, channels, height, width)
        - action (OrderedDict): The action with the highest duration in the clip (None if no actions are found)
        - start, end (tuple): Start and end frame index of the clip
        """
        if self.curr + self.wsize > self.nframes:
            raise StopIteration
        
        return self._sample()

    def __call__(self, probability: float = 1):
        if self.curr:
            raise Exception("Sampler is already being used as an iterator")
        if self.sample_limit and probability == 1:
            warnings.warn(f"Sample limit + 100% sampling probability is discouraged, as it reduces dataset usage", RuntimeWarning)
        
        samples = []        
        rng = np.random.default_rng()

        while self.curr + self.wsize < self.nframes:
            if self.sample_limit and len(samples) > self.sample_limit:
                break

            magic_number = rng.random()
            if magic_number > probability:
                self.curr += self.stride
                continue
            
            s = self._sample()
            samples.append(s)
            
        return samples
