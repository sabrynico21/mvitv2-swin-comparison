import os
from sys import stderr
import datasets
import cv2
import numpy as np
from itertools import cycle
from collections import OrderedDict
from torch.utils.data import IterableDataset
from remotezip import RemoteZip
from utils import timings_to_frames

class CharadesDataset(IterableDataset):
    def __init__(self, transform=None, shuffle_bufsize=1024, split="train"):
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
        self.videos = self.videos.shuffle(buffer_size=shuffle_bufsize).with_format("torch")
        self.transform = transform

    def __iter__(self):
        for video in cycle(self.videos):
            yield video
    
    def extract_sample(self, data):
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
        print(video_path, frame_rate, len(frames), file=stderr)
        frames = np.array(frames)

        if self.transform:
            frames = self.transform(frames)

        timings = timings_to_frames(video_info["timings"], frame_rate)

        sample = {
            'video': frames,
            'framerate': frame_rate,
            'objects': video_info["objects"],
            'actions': video_info["actions"],
            'timings': timings
        }

        return sample