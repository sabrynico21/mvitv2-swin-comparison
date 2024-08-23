import warnings
import cv2
import numpy as np
import torch
import constants
import pickle
from sys import stderr
from collections import OrderedDict
from typing import List
from torchvision import transforms

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

class ClipSampler:
    def __init__(self, data, stride_range: tuple = (6,12), max_samples: int = None, clip_size = 16, mode='dilated'):
        self._data = data
        self.mode = mode
        self.video = data.video
        self.framerate = data.framerate
        self.wsize = clip_size
        self.curr = 0
        self.sample_limit = max_samples
        self.nframes = self.video.size(0)
        if mode == 'dilated':
            self.desired_span = 3 # seconds
            self.dilation = round((self.framerate * self.desired_span) / (self.wsize - 1))
            if self.nframes < (self.wsize-1) * self.dilation:
                raise Exception("Insufficient video length. Skipping...")
        else:
            self.stride = round(min(stride_range[0] + self.framerate/2, stride_range[1] / (30.0/self.framerate)))
    def __iter__(self):
        self.curr = 0
        return self
    
    def _sample(self, single_action=False):
        # take the clip
        cstart = self.curr
        if self.mode == 'dilated':
            cend = cstart + (self.wsize-1) * self.dilation
            clip = self.video[cstart:cend+1:self.dilation]
            # print("cliplen=", len(clip), " framerate=", self.framerate, " dilation=", self.dilation, file=stderr)
        else:
            cend = cstart + self.wsize
            clip = self.video[cstart:cend]
            self.curr += self.stride

        # print("cstart=", cstart, " cend=", cend, " nframes=", self.nframes, file=stderr)
        
        clip_actions = OrderedDict()
        for tidx, action in enumerate(self._data.actions):
            tstart, tend = self._data.timings[tidx]
            if tstart <= cstart <= tend:
                action = str(action)
                # due to the actions merging into new classes, we need to check if the same action class was already added
                if clip_actions.get(constants.CLASS_MAPPING[action], None):
                    clip_actions[constants.CLASS_MAPPING[action]] += tend - cstart
                else:
                    clip_actions.update({constants.CLASS_MAPPING[action]: tend - cstart})
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
        
        rng = np.random.default_rng(42)

        if self.mode == 'dilated':
            self.curr = rng.integers(low=0, high=self.nframes - ((self.wsize-1) * self.dilation), size=None)
            return [self._sample()]
        
        samples = []        
        while self.curr + self.wsize < self.nframes:
            if self.sample_limit and len(samples) >= self.sample_limit:
                break
            magic_number = rng.random()
            if magic_number > probability:
                self.curr += self.stride
                continue
            
            s = self._sample()
            samples.append(s)
            
        return samples


def compute_action_frequencies():
    import pandas as pd

    df = pd.read_csv('Charades_v1_train.csv')

    num_classes = constants.NUM_CLASSES 
    action_freq = {i: 0 for i in range(num_classes)}
    print(len(constants.CLASS_MAPPING))
    # Process each row in the DataFrame
    for row, actions_str in enumerate(df['actions']):
        # Ensure actions_str is a string
        if isinstance(actions_str, str):
            # Split actions by ';'
            actions = actions_str.split(';')
            for action in actions:
                # Extract the action index, which is the part after 'c'
                if action:
                    action_index = str(int(action.split()[0][1:]))
                    print(action_index, df.loc[row]["id"])
                    action_freq[constants.CLASS_MAPPING[action_index]] += 1
        else:
            print(f"Skipping invalid value: {actions_str}")
            print(df.loc[row])
            continue
    # Convert frequencies to a dictionary
    action_freq = dict(action_freq)

    for action_index, freq in action_freq.items():
        print(f'Action index: {action_index}, Frequency: {freq}')

    with open('action_freq.pkl', 'wb') as pkl_file:
        pickle.dump(action_freq, pkl_file)

    return action_freq


def load_action_weights():
    try:
        with open('action_freq.pkl', 'rb') as file:
            action_freq_dict = pickle.load(file)
    except:
        print("Couldn't load action frequencies from file. Computing frequencies...", file=stderr)
        action_freq_dict = compute_action_frequencies()
    
    positive_counts= torch.tensor(np.array(list(action_freq_dict.values())))
    negative_counts = 7986 - positive_counts
    pos_weights = negative_counts / positive_counts

    return pos_weights
