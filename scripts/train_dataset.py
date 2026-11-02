import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class How2SignDataset(Dataset):
    def __init__(self, data_dir, config="full", max_frames=150):
        self.files = list(Path(data_dir).glob("*.npz"))
        self.config = config
        self.max_frames = max_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as data:
            left_hand = data["left_hand"]   # (T, 21, 3)
            right_hand = data["right_hand"] # (T, 21, 3)
            pose = data["pose"]             # (T, 25, 3) 
            face = data["face"]             # (T, 40, 3)
            text = str(data["text"].item())

        if self.config == "hands":
            features = np.concatenate([left_hand, right_hand], axis=2) 
        elif self.config == "hands_pose":
            features = np.concatenate([left_hand, right_hand, pose], axis=2) 
        else: # full
            features = np.concatenate([left_hand, right_hand, pose, face], axis=2) 

        features = features.reshape(features.shape[0], -1)
        
        T = features.shape[0]
        if T < self.max_frames:
            padding = np.zeros((self.max_frames - T, features.shape[1]))
            features = np.vstack((features, padding))
        else:
            features = features[:self.max_frames, :]

        return torch.tensor(features, dtype=torch.float32), text