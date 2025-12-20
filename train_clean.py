from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

ROOT_DIR = "faceforensics/data"
BATCH_SIZE = 16

# Modello ResNet50 senza pesi pretrained
model = resnet50(weights=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Unbalanced dataset since 4000 fake videos and 1000 real
# so balance fake vs real during training
train_counts = count_labels(train_dataset)
num_real_train = train_counts[0]
num_fake_train = train_counts[1]
pos_weight = torch.tensor([num_real_train/num_fake_train]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_clean(model, train_loader, val_loader, num_epochs, optimizer, device):
    train_losses = []
    train_metrics = Metrics()
    val_metrics = Metrics()

    for epoch in range(num_epochs):
        #TRAINING
        model.train()
        y_true, y_pred, y_prob = [], [], []
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            if batch is None:
                continue
            imgs, y = batch
            imgs, y = imgs.to(device), y.to(device).long()
            logits = model(imgs)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(int)

            y_true.extend(y.cpu().numpy().ravel())
            y_pred.extend(preds)
            y_prob.extend(probs)

        train_losses.append(train_loss/len(train_loader.dataset))
        train_metrics.compute(y_true, y_pred, y_prob)

        #VALIDATION
        model.eval()

        pbar = tqdm(val_loader, desc=f"Validation{epoch+1}/{num_epochs}", unit="batch")
        with torch.no_grad():
            for batch in pbar:
                if batch is None:
                    continue
                imgs, y = batch
                imgs, y = imgs.to(device), y.to(device).float().unsqueeze(1)
                logits = model(imgs)
    
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                preds = (probs >= 0.5).astype(int)
    
                y_true.extend(y.cpu().numpy().ravel())
                y_pred.extend(preds)
                y_prob.extend(probs)

        val_metrics.compute(y_true, y_pred, y_prob)

        print(f"Epoch {epoch+1}:")
        print("TRAINING")
        print("Training loss:", train_loss)
        train_metrics.print()
        print("VALIDATION")
        val_metrics.print()

        #TODO
        #SALVA LA STORIA DELLE METRICHE IN UN FILE JSON
        #SALVA I PESI DEL MODELLO

        #PLOT DELLA LOSS
