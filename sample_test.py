from torchvision import transforms
from dataset import FFDataset
from utils import balanced_subset, ROOT_DIR
import json

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
    
        
print("Initializing testing dataset....")
test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
# I get a subset of 1000 images (instead of 2800 total)
test_small, img_ids = balanced_subset(test_dataset, n_per_class=500)
img_ids = sorted(img_ids)

with open("sampled_test_set.json", "w") as f:
    f.write(json.dumps(img_ids, indent=2))


