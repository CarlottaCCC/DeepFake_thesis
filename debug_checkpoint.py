import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import *
import foolbox as fb
from train_robust import train_robust


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
seed=42
set_seed(42)
checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
model, criterion, optimizer, scheduler, checkpoint = reset_checkpoint_simple(checkpoint_path, device)

for key, val in checkpoint['model_state_dict'].items():
    if 'bn1.running_mean' in key:
        print(key, val[:5])
        break

# compare with loaded model
for m_name, m in model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        print(m_name, m.running_mean[:5])
        break

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
  ])

#print(train_dataset.getitem(0))
print("Initializing training dataset....")
train_dataset = FFDataset(root_dir=ROOT_DIR, split="train", transform=transform)
print("Initializing train loader...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Initializing validation dataset....")
val_dataset = FFDataset(root_dir=ROOT_DIR, split="val", transform=transform)
print("Initializing val loader....")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

imgs_train, y, _ = next(iter(train_loader))
imgs_val, y, _ = next(iter(val_loader))
print(imgs_train.min(), imgs_train.max(), imgs_train.mean())
print(imgs_val.min(), imgs_val.max(), imgs_val.mean()) 
#all_preds = []
#all_labels = []
#imgs, y, _ = next(iter(val_loader))
#imgs = imgs.to(device)
#
#model.eval()
#with torch.no_grad():
#    imgs, y, _ = next(iter(train_loader))
#    imgs = imgs.to(device)
#    out = model(normalize(imgs))
#    print(out)
#    print(torch.softmax(out, dim=1))

#print(f"Pred distribution: {sum(all_preds)} ones out of {len(all_preds)}")
#print(f"Label distribution: {sum(all_labels)} ones out of {len(all_labels)}")

train_metrics_clean = Metrics()
train_metrics_adv = Metrics()
val_metrics_clean = Metrics()
val_metrics_adv = Metrics()
start_epoch = 0
train_losses = []

# FGSM-AT + entropy penalty
train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses = train_robust(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader,
    epsilon=2/255,
    start_epoch=start_epoch, 
    num_epochs=12, 
    optimizer=optimizer,
    scheduler=scheduler, 
    criterion=criterion,
    device=device,
    train_losses=train_losses,
    train_metrics_clean=train_metrics_clean,
    train_metrics_adv=train_metrics_adv,
    val_metrics_clean=val_metrics_clean,
    val_metrics_adv=val_metrics_adv,
    seed=seed,
    sched_type="linear",
    lambda_entropy=0.01,
    entropy_flag=False,
    has_eps_sched=False,
    save_model=False)