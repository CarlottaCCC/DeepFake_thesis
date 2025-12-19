import os
from PIL import Image
from torch.utils.data import Dataset
import json

#reads the official splits json files of the FF++ dataset
def load_split(split_path):
    with open(split_path, 'r') as f:
        return json.load(f)

class FFDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None, manipulation="DeepFakes"):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(split_file, 'r') as f:
            video_ids = json.load(f)

        # REAL
        real_path = os.path.join(
            root_dir, "original_sequences", "c23", "frames"
        )
        for vid in video_ids:
            frame_dir = os.path.join(real_path, vid)
            if os.path.exists(frame_dir):
                for frame in os.listdir(frame_dir):
                    self.samples.append((os.path.join(frame_dir, frame), 0))

        # FAKE
        fake_path = os.path.join(
            root_dir, "manipulated_sequences", manipulation, "c23", "frames"
        )
        for vid in video_ids:
            frame_dir = os.path.join(fake_path, vid)
            if os.path.exists(frame_dir):
                for frame in os.listdir(frame_dir):
                    self.samples.append((os.path.join(frame_dir, frame), 1))

    def __len__(self):
        return len(self.samples)

    # 0:real , 1:fake
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label