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
from train_robust_curriculum import train_robust_with_curriculum
import foolbox as fb
import itertools
from test_attacks import test_attack
from train_robust_pgn import train_robust_with_curriculum_pgn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed=42
set_seed(42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
   ])
    
checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"

scheduler_type = ['linear']


param_grid = {
    'base_lr'        : [1e-6],
    'max_lr'         : [1e-4],
    'weight_decay'   : [1e-2],
    'batch_size'     : [32],
    'step_size_up'   : [1],        # in epochs
    'lambda_entropy' : [1e-4, 1e-3, 1e-2, 1e-1],
    'num_epochs_per_eps': [1],
    'lambda_grad': [1e-4, 1e-3, 1e-2, 1e-1]
}

best_score  = 0
best_params = {}
results     = []

# Generate all combinations
keys   = list(param_grid.keys())
values = list(param_grid.values())

for combo in itertools.product(*values):
    params = dict(zip(keys, combo))

    # Skip invalid combinations
    if params['base_lr'] >= params['max_lr']:
        continue

    print(f"\nTesting: {params}")
    torch.cuda.empty_cache()

    train_loader, val_loader, test_loader = get_data_loaders(params['batch_size'])

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
        num_epochs_rampup=10, type='linear',
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
    trained_model, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses = train_robust_with_curriculum_pgn(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        lambda_entropy=params['lambda_entropy'],
        lambda_grad=params['lambda_grad'],
        start_epoch=start_epoch, 
        num_epochs=10, 
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
        seed=seed,
        entropy_flag=True,
        adaptive=True,
        save_model=False)

    val_adv_acc = val_metrics_adv.accuracy_list[num_epochs-1]
    val_clean_acc = val_metrics_clean.accuracy_list[num_epochs-1]

    # testing over FGSM
    clean_metrics, fgsm_metrics_1 = test_attack(trained_model, test_loader, 'fgsm', 2/255, 'foolbox', " ", " ", "FGSM (eps=2/255)", device, save_results=False)
    clean_metrics, fgsm_metrics_2 = test_attack(trained_model, test_loader, 'fgsm', 8/255, 'foolbox', " ", " ", "FGSM (eps=8/255)", device, save_results=False)
    clean_metrics, ifgsm_metrics_2 = test_attack(trained_model, test_loader, 'ifgsm', 8/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
    clean_metrics, pgd_metrics_2 = test_attack(trained_model, test_loader, 'pgd', 8/255, 'foolbox', " ", " ", "PGD", device, save_results=False)


    if torch.isnan(torch.tensor(val_clean_acc)):
        results.append({
            'params': params, 
            'val_clean_acc': None, 
            'val_adv_acc':  None,
            'test_clean_acc':  None,
            'test_fgsm_small_acc':  None,
            'test_fgsm_big_acc':  None,
            'test_ifgsm_big_acc':  None,
            'test_pgd_big_acc':  None,
            'status': 'failed_nan'})
    else:
        results.append({
            'params': params, 
            'val_clean_acc': val_clean_acc, 
            'val_adv_acc': val_adv_acc,
            'test_clean_acc': clean_metrics.accuracy_list[0],
            'test_fgsm_small_acc': fgsm_metrics_1.accuracy_list[0],
            'test_fgsm_big_acc': fgsm_metrics_2.accuracy_list[0],
            'test_ifgsm_big_acc': ifgsm_metrics_2.accuracy_list[0],
            'test_pgd_big_acc': pgd_metrics_2.accuracy_list[0],
            'status': 'ok'})
    
    with open(f'results_grid_search/grid_search_params_pgn_with_test_results_linear_Cyclic_{num_epochs}_2.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Clean acc: {val_clean_acc:.4f} | Adv acc: {val_adv_acc:.4f}")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    
