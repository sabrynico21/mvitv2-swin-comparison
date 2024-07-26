import io
from sys import stderr
import datasets
import cv2
import numpy as np
from itertools import cycle
from collections import OrderedDict
import torch
from torch.utils.data import IterableDataset
from remotezip import RemoteZip
from utils import timings_to_frames

class CharadesDataset(IterableDataset):
    def __init__(self, transform=None):
        self.videos = datasets.load_dataset(
            "HuggingFaceM4/charades", 
            "480p", 
            split='train',
            streaming=True,
            trust_remote_code=True,
        )
        self.videos = self.videos.shuffle(buffer_size=1024).with_format("torch")
        self.transform = transform

    def __iter__(self):
        for video in self.videos:
            yield self.extract_sample(video)
    
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
        zip_url = video_path.split('::')[-1]
        video_name = video_path.split('::')[0].replace('zip://Charades_v1/', 'Charades_v1_480/')
        with RemoteZip(zip_url) as archive:
            video_data = archive.extract(video_name, "dataset")

        video_stream = cv2.VideoCapture(video_data)
        frames = []
        frame_rate = video_stream.get(cv2.CAP_PROP_FPS)

        while video_stream.isOpened():
            ret, frame = video_stream.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        video_stream.release()
        print(video_name, frame_rate, len(frames), file=stderr)
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
