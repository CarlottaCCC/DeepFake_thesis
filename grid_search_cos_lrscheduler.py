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
    #train_small, _ = balanced_subset(train_dataset, n_per_class=5)
    
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

#max_lr= 0.00026010272516552795,
#max_lr= 1e-6
#weight_decay = 0.005446894437507534
#step_size_up = 2
#lambda_entropy = 0.05328293739825308
#batch_size = 32

param_grid = {
    'base_lr'        : [1e-7, 1e-8],
    'weight_decay'   : [1e-4, 1e-2],
    'batch_size'     : [32],
    'step_size_up'   : [1],        # in epochs
    'lambda_entropy' : [1e-4, 1e-2]
}

best_score  = 0
best_params = {}
results     = []

# Generate all combinations
keys   = list(param_grid.keys())
values = list(param_grid.values())

for combo in itertools.product(*values):
    params = dict(zip(keys, combo))

    print(f"\nTesting: {params}")
    torch.cuda.empty_cache()

    train_loader, val_loader = get_data_loaders(params['batch_size'])

    checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"

    # Here I'm using the CosineAnnealingWarmRestarts lr scheduler,
    # it only needs base lr andstep_size_up
    model, criterion, optimizer, scheduler = reset_checkpoint(
        checkpoint_path=checkpoint_path, 
        sched_type="CosineAWR",
        device=device,
        base_lr=params['base_lr'], 
        max_lr=0, # does not need it now
        wd=params['weight_decay'],
        step_size_up=params['step_size_up'] * len(train_loader)
    )
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])  # split batches across GPUs
    model = model.to(device)

    eps_scheduler = CurriculumEpsilonScheduler(
        eps_start=0/255, eps_end=8/255,
        num_epochs_rampup=10, type='cosine',
        adaptive=True, patience=5, num_epochs_per_eps=1
    )

    train_metrics_clean = Metrics()
    train_metrics_adv = Metrics()
    val_metrics_clean = Metrics()
    val_metrics_adv = Metrics()
    start_epoch = 0
    num_epochs = eps_scheduler.num_epochs_per_eps * eps_scheduler.num_epochs_rampup
    train_losses = []
    
    # FGSM-AT + entropy penalty
    _ , _ , val_metrics_clean, val_metrics_adv, _ = train_robust_with_curriculum(
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

    val_adv_acc = val_metrics_adv.accuracy_list[num_epochs-1]
    val_clean_acc = val_metrics_clean.accuracy_list[num_epochs-1]

    # Track results
    results.append({
        'params': params, 
        'val_clean_acc': val_clean_acc, 
        'val_adv_acc': val_adv_acc })
    
    with open('grid_search_results_cosineEps_CosineAWR.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Clean acc: {val_clean_acc:.4f} | Adv acc: {val_adv_acc:.4f}")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    
