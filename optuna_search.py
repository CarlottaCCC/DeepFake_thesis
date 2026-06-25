import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch.nn as nn
from utils import *
from train_robust_curriculum import train_robust_with_curriculum
import foolbox as fb
import optuna

def init_dataloaders(batch_size):

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Initializing val loader....")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def objective(trial):

    device = torch.device('cuda')
    print(device)

    # Optuna suggests values — space is defined here, not upfront
    # LR
    #base_lr      = trial.suggest_float('base_lr', 1e-7, 1e-6, log=True)
    base_lr = 1e-7
    max_lr       = trial.suggest_float('max_lr', 1e-4, 1e-3, log=True)
    # WD
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    # CyclicLR specific
    step_size_up = trial.suggest_int('step_size_up', 1, 3)
    # LAMBDA for entropy penalty
    lambda_entropy = trial.suggest_float('lambda_entropy', 1e-3, 1e-1, log=True)
    # BATCH SIZE
    batch_size  = trial.suggest_categorical('batch_size', [16, 32])
    # NUM OF EPOCHS FOR EACH EPSILON
    #num_epochs_per_eps = trial.suggest_categorical('num_epochs_per_eps', [2, 3, 4, 5])

    if base_lr >= max_lr:
        raise optuna.exceptions.TrialPruned()
    
    # Prevent unstable combinations
    #  a high LR with a high entropy penalty can destabilize training quickly
    if max_lr > 1e-3 and lambda_entropy > 1e-2:
        raise optuna.exceptions.TrialPruned()

    eps_scheduler = CurriculumEpsilonScheduler(
        eps_start=0/255, eps_end=9/255,
        num_epochs_rampup=10, type='cosine',
        adaptive=True, patience=5, num_epochs_per_eps=1
    )

    # Init data loaders
    train_loader, val_loader = init_dataloaders(batch_size)

    checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
    model, criterion, optimizer, scheduler = reset_checkpoint(
        checkpoint_path=checkpoint_path, 
        base_lr=base_lr, 
        max_lr=max_lr, 
        wd=weight_decay,
        step_size_up=step_size_up * len(train_loader), 
        device=device)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])  # split batches across GPUs
    model = model.to(device)

    train_metrics_clean = Metrics()
    train_metrics_adv = Metrics()
    val_metrics_clean = Metrics()
    val_metrics_adv = Metrics()
    start_epoch = 0
    num_epochs = eps_scheduler.num_epochs_per_eps * eps_scheduler.num_epochs_rampup
    train_losses = []

    print("HYPERPARAMETERS:")
    print(f"Base lr: {base_lr}")
    print(f"Max lr: {max_lr}")
    print(f"Lambda entropy: {lambda_entropy}")
    print(f"Weight decay: {weight_decay}")
    print(f"batch size: {batch_size}")
    print(f"step size up: {step_size_up}")

    _ , _ , val_metrics_clean, val_metrics_adv, _ = train_robust_with_curriculum(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader,
    lambda_entropy=lambda_entropy,
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
    save_model=False)

    adv_val_accuracy = val_metrics_adv.accuracy_list[num_epochs-1]

    return adv_val_accuracy


if __name__ == "__main__":
    set_seed(42)

    study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=50)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    best = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number
    }
    
    with open("best_params_cosine_entropy_lr_1em7.json", "w") as f:
        json.dump(best, f, indent=4)
    
    # Reload later
    #with open("best_params.json", "r") as f:
    #    best = json.load(f)
    #    best_params = best["best_params"]