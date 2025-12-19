import os
from PIL import Image
from torch.utils.data import Dataset
import json
from torchvision import transforms


#reads the official splits json files of the FF++ dataset
def load_split(split_path):
    with open(split_path, 'r') as f:
        return json.load(f)

class FFDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, manipulation="DeepFakes"):
        self.root_dir = root_dir
        self.transform = transform
        self.split_file = f"faceforensics/splits/{split}.json"
        self.samples = []

        with open(self.split_file, 'r') as f:
            video_ids = json.load(f)

        # REAL
        real_path = os.path.join(
            root_dir, "original_sequences", "c23", "frames"
        )
        for vid in video_ids:
            # print(vid)
            vid1 = vid[0]
            vid2 = vid[1]
            frame_dir1 = os.path.join(real_path, vid1)
            frame_dir2 = os.path.join(real_path, vid2)
            if os.path.exists(frame_dir1):
                for frame in os.listdir(frame_dir1):
                    self.samples.append((os.path.join(frame_dir1, frame), 0))

            if os.path.exists(frame_dir2):
                for frame in os.listdir(frame_dir2):
                    self.samples.append((os.path.join(frame_dir2, frame), 0))

        # FAKE
        fake_path = os.path.join(
            root_dir, "manipulated_sequences", manipulation, "c23", "frames"
        )
        for vid in video_ids:
            vid1 = vid[0]
            vid2 = vid[1]
            frame_dir1 = os.path.join(fake_path, f"{vid1}_{vid2}")
            frame_dir2 = os.path.join(fake_path, f"{vid2}_{vid1}")
            if os.path.exists(frame_dir1):
                for frame in os.listdir(frame_dir1):
                    self.samples.append((os.path.join(frame_dir1, frame), 1))

            if os.path.exists(frame_dir2):
                for frame in os.listdir(frame_dir2):
                    self.samples.append((os.path.join(frame_dir2, frame), 1))

    def __len__(self):
        return len(self.samples)

    # 0:real , 1:fake
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
