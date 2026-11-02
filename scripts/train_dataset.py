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

        # Flatten the (landmarks, 3) dimension into a single feature vector per channel
        T = left_hand.shape[0]
        left_flat = left_hand.reshape(T, -1)   # (T, 63)
        right_flat = right_hand.reshape(T, -1) # (T, 63)
        pose_flat = pose.reshape(T, -1)        # (T, 75)
        face_flat = face.reshape(T, -1)        # (T, 120)

        # Concatenate the flattened arrays along the feature axis (axis=1)
        if self.config == "hands":
            features = np.concatenate([left_flat, right_flat], axis=1) # 126
        elif self.config == "hands_pose":
            features = np.concatenate([left_flat, right_flat, pose_flat], axis=1) # 201
        else: # full
            features = np.concatenate([left_flat, right_flat, pose_flat, face_flat], axis=1) # 321
        
        # Temporal padding to max_frames
        if T < self.max_frames:
            padding = np.zeros((self.max_frames - T, features.shape[1]))
            features = np.vstack((features, padding))
        else:
            features = features[:self.max_frames, :]

        return torch.tensor(features, dtype=torch.float32), text