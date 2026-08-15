import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from utils import *
import itertools
from test_attacks import test_attack
from diffusers import DDPMPipeline, DDPMScheduler
 
from ades import LearnableEpsilonScheduler, compute_adaptive_epsilon, pgd_attack_adaptive_eps
from difat import DiffusionPurifier, dpgd_attack


#model_name_list = [{"model_name":"resnet50_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_8.0__lr_0.001_seed_42_epochs_40_freeze.pt", "mode":"ades", "lambda_mean":8, "loss_type":"MAXLOSS_LINEAR_TARGET", "num_epochs":40},
#                   {"model_name":"resnet50_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_8.0__lr_0.001_seed_42_epochs_25_freeze.pt", "mode":"ades", "lambda_mean":8, "loss_type":"MAXLOSS_LINEAR_TARGET", "num_epochs":25},
#                   {"model_name":"resnet50_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_8.0__lr_0.001_seed_42_epochs_15_freeze.pt", "mode":"ades", "lambda_mean":8, "loss_type":"MAXLOSS_LINEAR_TARGET", "num_epochs":15},
#                   {"model_name":"resnet50_pgdat_baseline__linear_eps_sched_numeprampup20_lr_0.001_seed_42_epochs_40_freeze_2.pt", "mode":"baseline", "num_epochs":40},
#                   {"model_name":"resnet50_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25_freeze.pt", "mode":"baseline", "num_epochs":25}]
model_name_list = [{"model_name":"resnet50_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25_freeze.pt", "mode":"baseline", "num_epochs":25}]

#model_name_list = [{"model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt", "mode":"clean_mmodel", "lambda_mean":0, "loss_type":"None", "num_epochs":40}]


for values in model_name_list:
    model_name = values["model_name"]
    mode = values["mode"]
    num_epochs = values["num_epochs"]
    if mode == "ades":
        lambda_mean = values["lambda_mean"]
        loss_type = values["loss_type"]
    results = []
    
    lr = 1e-03
    weight_decay =5e-4
    batch_size = 32
    alpha_adv = 0.5
    sched_type = "linear"
    epsilon = 8/255
    
    device = torch.device("cuda")
    print(device)
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, 2)
    )
    #model = model.to(device)
    model = model.cuda()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
    
    
    #checkpoint_path = f"{MODELS_DIR}/pgdat_ades_difat/{model_name}"
    checkpoint_path = f"{MODELS_DIR}/pgdat_ades_difat/{model_name}"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Initializing Data loaders ......")
    train_loader, val_loader, test_loader = get_data_loaders(transform, batch_size)
    print(f"Starting testing {model_name}")
    
    # testing
    clean_metrics, fgsm_metrics_4 = test_attack(model, test_loader, 'fgsm', 4/255, 'foolbox', " ", " ", "FGSM (eps=4/255)", device, save_results=False)
    clean_metrics, ifgsm_metrics_4 = test_attack(model, test_loader, 'ifgsm', 4/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
    clean_metrics, pgd_metrics_4 = test_attack(model, test_loader, 'pgd', 4/255, 'foolbox', " ", " ", "PGD", device, save_results=False)
    
    clean_metrics, fgsm_metrics_1 = test_attack(model, test_loader, 'fgsm', 2/255, 'foolbox', " ", " ", "FGSM (eps=2/255)", device, save_results=False)
    clean_metrics, fgsm_metrics_2 = test_attack(model, test_loader, 'fgsm', 8/255, 'foolbox', " ", " ", "FGSM (eps=8/255)", device, save_results=False)
    clean_metrics, ifgsm_metrics_2 = test_attack(model, test_loader, 'ifgsm', 8/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
    clean_metrics, pgd_metrics_2 = test_attack(model, test_loader, 'pgd', 8/255, 'foolbox', " ", " ", "PGD", device, save_results=False)
    clean_metrics, ifgsm_metrics_1 = test_attack(model, test_loader, 'ifgsm', 2/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
    clean_metrics, pgd_metrics_1 = test_attack(model, test_loader, 'pgd', 2/255, 'foolbox', " ", " ", "PGD", device, save_results=False)
    
    results.append({
        'test_clean_acc': clean_metrics.accuracy_list[0],
        'test_clean_auc': clean_metrics.auc_list[0],
        'test_clean_asr': 0,
        'test_fgsm_small_acc': fgsm_metrics_1.accuracy_list[0],
        'test_fgsm_small_auc': fgsm_metrics_1.auc_list[0],
        'test_fgsm_small_asr': fgsm_metrics_1.asr_list[0],
        'test_fgsm_med_acc': fgsm_metrics_4.accuracy_list[0],
        'test_fgsm_med_auc': fgsm_metrics_4.auc_list[0],
        'test_fgsm_med_asr': fgsm_metrics_4.asr_list[0],
        'test_fgsm_big_acc': fgsm_metrics_2.accuracy_list[0],
        'test_fgsm_big_auc': fgsm_metrics_2.auc_list[0],
        'test_fgsm_big_asr': fgsm_metrics_2.asr_list[0],
        'test_ifgsm_small_acc': ifgsm_metrics_1.accuracy_list[0],
        'test_ifgsm_small_auc': ifgsm_metrics_1.auc_list[0],
        'test_ifgsm_small_asr': ifgsm_metrics_1.asr_list[0],
        'test_ifgsm_med_acc': ifgsm_metrics_4.accuracy_list[0],
        'test_ifgsm_med_auc': ifgsm_metrics_4.auc_list[0],
        'test_ifgsm_med_asr': ifgsm_metrics_4.asr_list[0],
        'test_ifgsm_big_acc': ifgsm_metrics_2.accuracy_list[0],
        'test_ifgsm_big_auc': ifgsm_metrics_2.auc_list[0],
        'test_ifgsm_big_asr': ifgsm_metrics_2.asr_list[0],
        'test_pgd_small_acc': pgd_metrics_1.accuracy_list[0],
        'test_pgd_small_auc': pgd_metrics_1.auc_list[0],
        'test_pgd_small_asr': pgd_metrics_1.asr_list[0],
        'test_pgd_med_acc': pgd_metrics_4.accuracy_list[0],
        'test_pgd_med_auc': pgd_metrics_4.auc_list[0],
        'test_pgd_med_asr': pgd_metrics_4.asr_list[0],
        'test_pgd_big_acc': pgd_metrics_2.accuracy_list[0],
        'test_pgd_big_auc': pgd_metrics_2.auc_list[0],
        'test_pgd_big_asr': pgd_metrics_2.asr_list[0],
        'status': 'ok'})
        
    out_dir = f'pgd_{mode}'

    os.makedirs(out_dir, exist_ok=True)
    eps_sched_label = "_"
    if mode == "baseline" and sched_type != "":
        eps_sched_label = f"linear_eps_sched"
    elif mode == "baseline" and sched_type == "":
        eps_sched_label = f"_fixed_eps_{epsilon}"
    elif mode == "ades":
        mode = f"ades_lambda_mean_{lambda_mean}_{loss_type}"
    with open(f'{out_dir}/test_pgd_{mode}_{eps_sched_label}_lr_{lr}_alpha_{alpha_adv}_num_epochs_{num_epochs}.json', 'w') as f:
        json.dump(results, f, indent=4)