from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm

ROOT_DIR = "faceforensics/data"
BATCH_SIZE = 16

# Modello ResNet50 senza pesi pretrained
model = resnet50(weights=None)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
   ])
    
train_dataset = FFDataset(root_dir=ROOT_DIR, split="train", transform=transform)
#print(train_dataset.getitem(0))
val_dataset = FFDataset(root_dir=ROOT_DIR, split="val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_clean(model, evaluator, train_loader, val_loader, num_epochs, optimizer, device):
    train_losses = []
    train_accuracy = []
    train_recall = []
    #ROC e AUROC

    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            logits = model(batch)
