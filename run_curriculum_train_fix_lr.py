import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import *
import foolbox as fb
from train_robust import train_robust

def get_data_loaders(batch_size):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])

    print("Initializing training dataset....")
    train_dataset = FFDataset(root_dir=ROOT_DIR, split="train", transform=transform)
    # I get a small subset for debugging
    #train_small, _ = balanced_subset(train_dataset, n_per_class=5)
    
    #print(train_dataset.getitem(0))
    print("Initializing validation dataset....")
    val_dataset = FFDataset(root_dir=ROOT_DIR, split="val", transform=transform)
    
    print("Initializing train loader...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Initializing val loader....")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    seed=42
    set_seed(42)

    base_lr = 1e-07
    #max_lr = 1e-05
    weight_decay = 0.0001
    batch_size = 32
    step_size_up = 1
    lambda_entropy = 0.0001

    train_loader, val_loader = get_data_loaders(batch_size)

    checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
    model, criterion, optimizer, scheduler = reset_checkpoint(
        checkpoint_path=checkpoint_path, 
        sched_type="CyclicLR",
        device=device,
        base_lr=base_lr
        )
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])  # split batches across GPUs
    model = model.to(device)

    #eps_scheduler = CurriculumEpsilonScheduler(
    #    eps_start=0/255, eps_end=8/255,
    #    num_epochs_rampup=10, type='cosine',
    #    adaptive=True, patience=5, num_epochs_per_eps=1
    #)

    train_metrics_clean = Metrics()
    train_metrics_adv = Metrics()
    val_metrics_clean = Metrics()
    val_metrics_adv = Metrics()
    start_epoch = 0
    num_epochs = 10
    train_losses = []
    
    # FGSM-AT + entropy penalty
    train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses = train_robust(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        epsilon=0,
        start_epoch=start_epoch, 
        num_epochs=num_epochs, 
        optimizer=optimizer,
        scheduler=None, 
        criterion=criterion,
        device=device,
        train_losses=train_losses,
        train_metrics_clean=train_metrics_clean,
        train_metrics_adv=train_metrics_adv,
        val_metrics_clean=val_metrics_clean,
        val_metrics_adv=val_metrics_adv,
        seed=seed,
        sched_type="cosine",
        lambda_entropy=lambda_entropy,
        entropy_flag=True,
        has_eps_sched=True,
        save_model=False)

