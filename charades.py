import datasets
import cv2
import numpy as np
import os
import torch
import constants
from dataclasses import dataclass
from sys import stderr
from typing import List
from itertools import cycle
from collections import OrderedDict
from torch.utils.data import IterableDataset
from remotezip import RemoteZip

@dataclass
class CharadesSample:
    video: torch.Tensor
    framerate: float
    objects: List[str]
    actions: List[int]
    timings: List[int]

class CharadesDataset(IterableDataset):
    def __init__(self, transform=None, shuffle_bufsize=128, split="train"):
        if split not in ["train", "test"]:
            raise ValueError("Invalid dataset split specified. Choices: train, test")
        
        self.url = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
        self.zip_ref = RemoteZip(self.url)
        self.videos = datasets.load_dataset(
            "HuggingFaceM4/charades", 
            "480p", 
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        self.videos = self.videos.shuffle(buffer_size=shuffle_bufsize)
        self.videos = self.videos.take(128*constants.BATCH_SIZE).with_format("torch")
        self.transform = transform

    def __iter__(self):
        for video in self.videos:
            yield video

    def _convert_timings_to_frames(self, ts, frame_rate):
        output = []
        for t in ts:
            output.append([
                torch.round(t[0] * frame_rate), 
                torch.round(t[1] * frame_rate)
            ])
        return output

    def extract_sample(self, data) -> CharadesSample:
        video_info = OrderedDict(
            video_id=data["video_id"],
            scene=data["scene"],
            verified=data["verified"],
            actions=data["labels"],
            objects=data["objects"],
            timings=data["action_timings"],
            length=data["length"],
        )

        video_path = data["video"]
        video_file = video_path.split('::')[0].replace('zip://Charades_v1/', 'Charades_v1_480/')
        video_path = os.path.join("dataset", video_file)
        if not os.path.isfile(video_path):
            video_file = self.zip_ref.extract(video_file, "dataset")

        video_stream = cv2.VideoCapture(video_path)
        frames = []
        frame_rate = video_stream.get(cv2.CAP_PROP_FPS)

        while video_stream.isOpened():
            ret, frame = video_stream.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        video_stream.release()
        # print(video_path, frame_rate, len(frames), file=stderr)
        frames = np.array(frames)

        if self.transform:
            frames = self.transform(frames)

        timings = self._convert_timings_to_frames(video_info["timings"], frame_rate)

        sample = CharadesSample(
            frames, 
            frame_rate, 
            video_info['objects'], 
            video_info['actions'], 
            timings
        )

        return sample