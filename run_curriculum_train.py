import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import *
from train_robust_curriculum import train_robust_with_curriculum
import foolbox as fb
import itertools

def get_data_loaders(batch_size):
    print("Initializing training dataset....")
    train_dataset = FFDataset(root_dir=ROOT_DIR, split="train", transform=transform)
    # I get a small subset for debugging
    #train_small, _ = balanced_subset(train_dataset, n_per_class=32)
    
    #print(train_dataset.getitem(0))
    print("Initializing validation dataset....")
    val_dataset = FFDataset(root_dir=ROOT_DIR, split="val", transform=transform)
    
    print("Initializing train loader...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Initializing val loader....")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed=42
set_seed(42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
   ])
    
checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"

scheduler_type = ['linear', 'cosine']


params = {"base_lr": 1e-07,
"max_lr": 1e-05,
"weight_decay": 0.0001,
"batch_size": 32,
"step_size_up": 1,
"lambda_entropy": 0.0001,
"num_epochs_per_eps": 1
}

sched_types = ['random']

for type in sched_types:

    print(f"\nTesting: {params}")
    torch.cuda.empty_cache()

    train_loader, val_loader = get_data_loaders(params['batch_size'])

    checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
    model, criterion, optimizer, scheduler = reset_checkpoint(
        checkpoint_path=checkpoint_path, 
        sched_type="CyclicLR",
        device=device,
        base_lr=params['base_lr'], 
        max_lr=params['max_lr'], 
        wd=params['weight_decay'],
        step_size_up=params['step_size_up'] * len(train_loader), 
        )
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])  # split batches across GPUs
    model = model.to(device)

    eps_scheduler = CurriculumEpsilonScheduler(
        eps_start=0/255, eps_end=8/255,
        num_epochs_rampup=10, type=type,
        adaptive=True, patience=5, num_epochs_per_eps=params['num_epochs_per_eps']
    )

    train_metrics_clean = Metrics()
    train_metrics_adv = Metrics()
    val_metrics_clean = Metrics()
    val_metrics_adv = Metrics()
    start_epoch = 0
    num_epochs = eps_scheduler.num_epochs_per_eps * eps_scheduler.num_epochs_rampup
    train_losses = []
    
    # FGSM-AT + entropy penalty
    train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, _ = train_robust_with_curriculum(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader,
    lambda_entropy=params['lambda_entropy'],
    start_epoch=start_epoch, 
    num_epochs=num_epochs, 
    optimizer=optimizer,
    scheduler=scheduler, 
    eps_scheduler=eps_scheduler,
    criterion=criterion,
    device=device,
    train_losses=train_losses,
    train_metrics_clean=train_metrics_clean,
    train_metrics_adv=train_metrics_adv,
    val_metrics_clean=val_metrics_clean,
    val_metrics_adv=val_metrics_adv,
    seed=42,
    entropy_flag=True,
    adaptive=True,
    save_model=True)
    
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    train_loader, val_loader = get_data_loaders(params['batch_size'])

    checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
    model, criterion, optimizer, scheduler = reset_checkpoint(
        checkpoint_path=checkpoint_path, 
        sched_type="CyclicLR",
        device=device,
        base_lr=params['base_lr'], 
        max_lr=params['max_lr'], 
        wd=params['weight_decay'],
        step_size_up=params['step_size_up'] * len(train_loader), 
        )
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])  # split batches across GPUs
    model = model.to(device)

    eps_scheduler = CurriculumEpsilonScheduler(
        eps_start=0/255, eps_end=8/255,
        num_epochs_rampup=10, type=type,
        adaptive=True, patience=5, num_epochs_per_eps=params['num_epochs_per_eps']
    )

    train_metrics_clean = Metrics()
    train_metrics_adv = Metrics()
    val_metrics_clean = Metrics()
    val_metrics_adv = Metrics()
    start_epoch = 0
    num_epochs = eps_scheduler.num_epochs_per_eps * eps_scheduler.num_epochs_rampup
    train_losses = []
    
    # FGSM-AT NO entropy
    train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, _ = train_robust_with_curriculum(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader,
    lambda_entropy=params['lambda_entropy'],
    start_epoch=start_epoch, 
    num_epochs=num_epochs, 
    optimizer=optimizer,
    scheduler=scheduler, 
    eps_scheduler=eps_scheduler,
    criterion=criterion,
    device=device,
    train_losses=train_losses,
    train_metrics_clean=train_metrics_clean,
    train_metrics_adv=train_metrics_adv,
    val_metrics_clean=val_metrics_clean,
    val_metrics_adv=val_metrics_adv,
    seed=42,
    entropy_flag=False,
    adaptive=True,
    save_model=True)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()



    
