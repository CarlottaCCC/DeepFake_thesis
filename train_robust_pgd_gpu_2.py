import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import *
import foolbox as fb
from test_attacks import test_attack
import itertools
from train_robust_pgd_gpu_1 import train_robust_PGD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed=42
set_seed(42)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
   ])
param_grid = {"base_lr": [1e-06],
"max_lr": [1e-04],
"weight_decay": [0.01],
"batch_size": [16],
"step_size_up": [1],
"lambda_entropy": [0.001, 0.005, 0.01],
"num_epochs_per_eps": [1],
"lambda_grad": [0.01]
}
# Generate all combinations
keys   = list(param_grid.keys())
values = list(param_grid.values())

# TRAINING WITH FGSM AUX LOSS
for combo in itertools.product(*values):
    params = dict(zip(keys, combo))
    print(f"\nTesting: {params}")

    torch.cuda.empty_cache()

    results     = []

    print("Initializing Data loaders ......")
    train_loader, val_loader, test_loader = get_data_loaders(transform, params['batch_size'])

    checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"

    print(f"Start training PGD-AT")
    model, criterion, optimizer, scheduler = reset_checkpoint(
        checkpoint_path=checkpoint_path, 
        sched_type="CyclicLR",
        base_lr=params['base_lr'], 
        max_lr=params['max_lr'], 
        wd=params['weight_decay'],
        step_size_up= params['num_epochs_per_eps'] * len(train_loader), 
        device=device)
    
    train_metrics_clean = Metrics()
    train_metrics_adv = Metrics()
    val_metrics_clean = Metrics()
    val_metrics_adv = Metrics()
    start_epoch = 0
    train_losses = []

    # PARAMETERS #################
    num_epochs = 15
    alpha_adv = 0.5
    epsilon = 2/255
    entropy_flag = True
    has_eps_scheduler = True
    has_gradient_penalty = True
    has_fgsm_aux_loss = False
    lambda_fgsm = 0

    # EPSILON SCHEDULER
    type = 'linear'
    eps_scheduler = CurriculumEpsilonScheduler(
            eps_start=0/255, eps_end=8/255,
            num_epochs_rampup=num_epochs, type=type,
            adaptive=True, patience=5, num_epochs_per_eps=params['num_epochs_per_eps']
        )
    ##############################

    training = ""
    attack = ""
    if entropy_flag == True:
        training = "PGD-AT + entropy"
        attack = f"pgd_entropy_{params['lambda_entropy']}"
    else:
        training = "PGD-AT"
        attack = "pgd"

    if has_eps_scheduler == True:
        eps_label = f"eps_sched_{eps_scheduler.type}"
    else:
        eps_label = f"epsilon_{epsilon}"

    grad_penalty = "_"
    if has_gradient_penalty == True:
        grad_penalty = f"_with_pgn_lambda_{params['lambda_grad']}"

    fgsm_aux_loss = "_"
    if has_fgsm_aux_loss == True:
        fgsm_aux_loss = f"_with_fgsm_aux_{lambda_fgsm}"

    
    # FGSM-AT
    print(f"Starting training with FGSM AUX LOSS lambda_fgsm:{lambda_fgsm}")
    print(f"has_gradient_penalty: {has_gradient_penalty}, {params['lambda_grad']}")
    
    trained_model, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses = train_robust_PGD(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader,
    lambda_entropy=params["lambda_entropy"],
    lambda_grad=params["lambda_grad"],
    lambda_fgsm = lambda_fgsm,
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
    seed=seed,
    base_lr=params["base_lr"],
    max_lr=params["max_lr"],
    alpha_adv=alpha_adv,
    epsilon=epsilon,
    entropy_flag=entropy_flag,
    has_eps_scheduler=has_eps_scheduler,
    has_gradient_penalty=has_gradient_penalty,
    has_fgsm_aux_loss = has_fgsm_aux_loss,
    save_model=False)

    val_adv_acc = val_metrics_adv.accuracy_list[num_epochs-1]
    val_clean_acc = val_metrics_clean.accuracy_list[num_epochs-1]
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
     # testing
    clean_metrics, fgsm_metrics_1 = test_attack(trained_model, test_loader, 'fgsm', 2/255, 'foolbox', " ", " ", "FGSM (eps=2/255)", device, save_results=False)
    clean_metrics, fgsm_metrics_2 = test_attack(trained_model, test_loader, 'fgsm', 8/255, 'foolbox', " ", " ", "FGSM (eps=8/255)", device, save_results=False)
    clean_metrics, ifgsm_metrics_2 = test_attack(trained_model, test_loader, 'ifgsm', 8/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
    clean_metrics, pgd_metrics_2 = test_attack(trained_model, test_loader, 'pgd', 8/255, 'foolbox', " ", " ", "PGD", device, save_results=False)
    clean_metrics, ifgsm_metrics_1 = test_attack(trained_model, test_loader, 'ifgsm', 2/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
    clean_metrics, pgd_metrics_1 = test_attack(trained_model, test_loader, 'pgd', 2/255, 'foolbox', " ", " ", "PGD", device, save_results=False)


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
            'test_clean_auc': clean_metrics.auc_list[0],
            'test_clean_asr': 0,
            'test_fgsm_small_acc': fgsm_metrics_1.accuracy_list[0],
            'test_fgsm_small_auc': fgsm_metrics_1.auc_list[0],
            'test_fgsm_small_asr': fgsm_metrics_1.asr_list[0],
            'test_fgsm_big_acc': fgsm_metrics_2.accuracy_list[0],
            'test_fgsm_big_auc': fgsm_metrics_2.auc_list[0],
            'test_fgsm_big_asr': fgsm_metrics_2.asr_list[0],
            'test_ifgsm_small_acc': ifgsm_metrics_1.accuracy_list[0],
            'test_ifgsm_small_auc': ifgsm_metrics_1.auc_list[0],
            'test_ifgsm_small_asr': ifgsm_metrics_1.asr_list[0],
            'test_ifgsm_big_acc': ifgsm_metrics_2.accuracy_list[0],
            'test_ifgsm_big_auc': ifgsm_metrics_2.auc_list[0],
            'test_ifgsm_big_asr': ifgsm_metrics_2.asr_list[0],
            'test_pgd_small_acc': pgd_metrics_1.accuracy_list[0],
            'test_pgd_small_auc': pgd_metrics_1.auc_list[0],
            'test_pgd_small_asr': pgd_metrics_1.asr_list[0],
            'test_pgd_big_acc': pgd_metrics_2.accuracy_list[0],
            'test_pgd_big_auc': pgd_metrics_2.auc_list[0],
            'test_pgd_big_asr': pgd_metrics_2.asr_list[0],
            'alpha_adv': alpha_adv,
            'eps_scheduler': eps_scheduler.type,
            'training': training,
            'pgn': f'{grad_penalty}',
            'fgsm_aux_loss': f'{fgsm_aux_loss}',
            'status': 'ok'})
    
    with open(f'results_grid_search/pgd_training/grid_search_{attack}{grad_penalty}_{eps_label}_with_test_results_Cyclic_{num_epochs}_alpha_{alpha_adv}_{fgsm_aux_loss}.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Clean acc: {val_clean_acc:.4f} | Adv acc: {val_adv_acc:.4f}")
