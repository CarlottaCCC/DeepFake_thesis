import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import json
import os
import torch
import random
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Subset
from torchvision import transforms
from utils import *
from attacks import standard_pgd_attack
import foolbox as fb
from attacks_implementation.ifgsm import ifgsm_attack
from art.attacks.evasion import (
    SaliencyMapMethod,      # JSMA
    SquareAttack,           # Square Attack
    ZooAttack              # ZOO
)
import torchattacks #in the DiFAT paper they get TI-FGSM, MI-FGSM and DI-FGSM from this library

def test_transferability(model, surrogate, attack_type, epsilon, test_loader, device):

    clean_metrics = Metrics()
    adv_metrics = Metrics()
    total_samples = 0
    correct_adv = 0
    all_probs_clean_split = []
    
    model.eval()
    surrogate.eval()

    preprocessing = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        axis=-3  # per PyTorch (C, H, W)
        )
    fmodel = fb.PyTorchModel(surrogate, bounds=(0,1), preprocessing=preprocessing, device=device)

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters()),
        input_shape=(3, 224, 224),
        nb_classes=2,
        clip_values=(0.0, 1.0),
        preprocessing=(
            [0.485, 0.456, 0.406],  # mean ImageNet
            [0.229, 0.224, 0.225]   # std ImageNet
        ),
        device_type=device
        )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

    attack_fn = None

    if attack_type == 'fgsm':
        attack_fn = fb.attacks.FGSM()

    elif attack_type == 'ti-fgsm':
        attack_fn = torchattacks.TIFGSM(
        surrogate, eps=epsilon, alpha=2/255, steps=10, decay=0.0,
        kernel_name='gaussian', len_kernel=15, nsig=3,
        resize_rate=0.9, diversity_prob=0.5,
        )

    elif attack_type == 'mi-fgsm':
        attack_fn = torchattacks.MIFGSM(
        surrogate, eps=epsilon, alpha=2/255, steps=10, decay=1.0,
        )

    elif attack_type == 'di-fgsm':
        attack_fn = torchattacks.DIFGSM(
            surrogate, eps=epsilon, alpha=2/255, steps=10, decay=0.0,
            resize_rate=0.9, diversity_prob=0.5,
        )


    print(f"Starting {attack_type} test!")
    pbar = tqdm(test_loader, desc=f"Testing", unit="batch")
    for batch in pbar:
        if batch is None:
            continue
        imgs, labels, _ = batch
        imgs, labels = imgs.to(device), labels.to(device).long().squeeze()
        batch_size = imgs.size(0)

        norm_imgs = normalize(imgs)
        with torch.no_grad():
            clean_preds = model(norm_imgs).argmax(1)
        clean_correct_mask = (clean_preds == labels)
    
        # Only attack examples the model already gets right
        imgs_correct = imgs[clean_correct_mask]
        labels_correct = labels[clean_correct_mask]

        imgs_incorrect = imgs[~clean_correct_mask]
        labels_incorrect = labels[~clean_correct_mask]

        if attack_type == 'fgsm':
            _, imgs_adv, _ = attack_fn(fmodel, imgs_correct, labels_correct, epsilons=epsilon)
            if imgs_incorrect.shape[0] > 0:
                _, imgs_adv_other, _ = attack_fn(fmodel, imgs_incorrect, labels_incorrect, epsilons=epsilon)
            else:
                imgs_adv_other = imgs_incorrect  # empty tensor

        elif attack_type == 'pgd':
            imgs_adv = standard_pgd_attack(surrogate, imgs_correct.detach(), labels_correct, epsilon, alpha=2/255, steps=20, normalize=normalize)
            if imgs_incorrect.shape[0] > 0:
                imgs_adv_other = standard_pgd_attack(surrogate, imgs_incorrect.detach(), labels_incorrect, epsilon, alpha=2/255, steps=20, normalize=normalize)
            else:
                imgs_adv_other = imgs_incorrect  # empty tensor

        elif attack_type == 'ti-fgsm' or 'mi-fgsm' or 'di-fgsm':
            imgs_adv = attack_fn(imgs_correct, labels_correct)
            if imgs_incorrect.shape[0] > 0:
                imgs_adv_other = attack_fn(imgs_incorrect, labels_incorrect)
            else:
                imgs_adv_other = imgs_incorrect  # empty tensor

        imgs_adv = normalize(imgs_adv.detach())
        imgs_adv_other = normalize(imgs_adv_other.detach())
        imgs = normalize(imgs.detach())

        # compute L2 and Linf metrics
        delta = imgs_adv - imgs_correct
        l2, linf = batch_norms(delta)
        adv_metrics.total_l2 += l2.sum().item()
        adv_metrics.total_linf += linf.sum().item()

        # inferenza
        with torch.no_grad():
            # HERE I USE THE TARGET MODEL TO CLASSIFY THE 
            # ADV IMAGES GENERATED BY THE SURROGATE MODEL
            logits_clean = model(imgs)
            logits_adv = model(imgs_adv)
            logits_adv_other = None
            if imgs_incorrect.shape[0] > 0:
                logits_adv_other = model(imgs_adv_other)

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1]

            if imgs_incorrect.shape[0] > 0:
                probs_clean_split = torch.cat([
                    probs_clean[clean_correct_mask],
                    probs_clean[~clean_correct_mask]
                ])

            else:
                probs_clean_split = probs_clean

            probs_clean = probs_clean.detach().cpu().numpy()
            probs_clean_split = probs_clean_split.detach().cpu().numpy()
            all_probs_clean_split.append(probs_clean_split)
            #debug
            preds_adv = torch.argmax(logits_adv, dim=1)
            clean_metrics.update(labels, probs_clean)
            probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
            #probs_adv_other = torch.softmax(logits_adv_other, dim=1)[:, 1].cpu().numpy()

            if imgs_incorrect.shape[0] > 0:
                all_probs_adv = torch.cat([torch.softmax(logits_adv, dim=1)[:,1],
                                            torch.softmax(logits_adv_other, dim=1)[:, 1]])
                all_probs_adv = all_probs_adv.detach().cpu().numpy()
                all_labels_adv = torch.cat([
                    labels_correct,
                    labels_incorrect
                ])
            else:
                all_probs_adv = probs_adv
                all_labels_adv = labels_correct

            adv_metrics.update(all_labels_adv, all_probs_adv)
        correct_adv += (preds_adv == labels_correct).sum().item()

        total_samples += batch_size
            
    #Attack success rate
    #adv_metrics.attack_success_rate(clean_metrics.all_probs)
    adv_metrics.attack_success_rate(all_probs_clean_split)
    # Average L2 and L_inf norm
    adv_metrics.avg_l2 = adv_metrics.total_l2/total_samples
    adv_metrics.avg_linf = adv_metrics.total_linf/total_samples
    
    clean_results = clean_metrics.compute()
    adv_results = adv_metrics.compute()
    robust_acc = 100 * correct_adv / total_samples
    adv_metrics.correct_adv_accuracy = robust_acc
    
    print("CLEAN RESULTS")
    clean_metrics.print(0)
    print(f"Correct adv accuracy: {adv_metrics.correct_adv_accuracy}")
    print(f"{attack_type} ATTACK RESULTS")
    adv_metrics.print(0)
    print(f"Attack Success Rate:  {adv_metrics.asr_list[0]}")
    print(f"Average L2:  {adv_metrics.avg_l2}")
    print(f"Average Linf:  {adv_metrics.avg_linf}")

    return clean_metrics, adv_metrics

if __name__ == "__main__":

    seed=42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Modello ResNet50 senza pesi pretrained - TARGET MODEL
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)

    # ResNet18 - SURROGATE MODEL
    surrogate = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    surrogate.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(surrogate.fc.in_features, 2)
    )
    surrogate = surrogate.to(device)

    checkpoint_path_surrogate = f"{MODELS_DIR}/resnet18/resnet18_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25.pt"
    #checkpoint_path_surrogate = f"{MODELS_DIR}/resnet18/resnet18_clean_epoch_12_lr_0.0001_wd_0.01_aug.pt"
    checkpoint_surrogate = torch.load(checkpoint_path_surrogate, map_location=device, weights_only=False)
    surrogate.load_state_dict(checkpoint_surrogate['model_state_dict'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader, val_loader, test_loader = get_data_loaders(transform, 32)

    models_names = [{"model_label":"PGD-AT linear eps sched", "model_name":"pgdat_ades_difat/resnet50_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25_freeze.pt"},
                            {"model_label":"ADES", "model_name":"pgdat_ades_difat/resnet50_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_50__lr_0.001_seed_42_epochs_25_freeze_norampup.pt"},
                            {"model_label":"ADES + 11 epochs PGD-AT", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_ades_baseline_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"PGD-AT + DiFat conf 2", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_7__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"PGD-AT + DiFat conf 4", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_[3, 6, 13]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"ADES + PGD-AT + DiFat", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_ades_baseline_difat_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23]_epochsdifat_[24]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"baseline model", "model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"}]

    #models_names = [{"model_label":"baseline model", "model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"}]
    out_dir = "test_transferability_results/surrogate_clean"
    os.makedirs(out_dir, exist_ok=True)

    epsilon_list = [8/255]

    for model_item in models_names:
        results = []
   
        model_label = model_item["model_label"]
        model_name = model_item["model_name"]
        checkpoint_path = f"{MODELS_DIR}/{model_name}"
        model = reset_model(checkpoint_path,device)

        for epsilon in epsilon_list:
            eps = epsilon * 255

            path_name = f"test_transferability_results/surrogate_2/{model_label}_transferability_results_eps_{eps}_v2.json"
    
            print(f"START TESTING TRANSFERABILITY ROBUSTNESS FOR MODEL {model_label}")
    
            clean_metrics, fgsm_metrics = test_transferability(model, surrogate, 'fgsm', epsilon, test_loader, device)
            clean_metrics, pgd_metrics = test_transferability(model, surrogate, 'pgd', epsilon, test_loader, device)
            clean_metrics, ti_fgsm_metrics = test_transferability(model, surrogate, 'ti-fgsm', epsilon, test_loader, device)
            clean_metrics, di_fgsm_metrics = test_transferability(model, surrogate, 'di-fgsm', epsilon, test_loader, device)
            clean_metrics, mi_fgsm_metrics = test_transferability(model, surrogate, 'mi-fgsm', epsilon, test_loader, device)
    
            results.append({
                    'test_clean_acc': clean_metrics.accuracy_list[0],
                    'test_clean_auc': clean_metrics.auc_list[0],
                    'test_clean_asr': 0,
                    'test_fgsm_acc': fgsm_metrics.accuracy_list[0],
                    'fgsm_correct_adv': fgsm_metrics.correct_adv_accuracy,
                    'test_fgsm_auc': fgsm_metrics.auc_list[0],
                    'test_fgsm_asr': fgsm_metrics.asr_list[0],
                    'fgsm_l2': fgsm_metrics.avg_l2,
                    'fgsm_linf': fgsm_metrics.avg_linf,
        
                    'test_ti-fgsm_acc': ti_fgsm_metrics.accuracy_list[0],
                    'test_ti-fgsm_auc': ti_fgsm_metrics.auc_list[0],
                    'test_ti-fgsm_asr': ti_fgsm_metrics.asr_list[0],
                    'ti-fgsm_correct_adv': ti_fgsm_metrics.correct_adv_accuracy,
                    'ti-fgsm_l2': ti_fgsm_metrics.avg_l2,
                    'ti-fgsm_linf': ti_fgsm_metrics.avg_linf,
        
                    'test_di-fgsm_acc': di_fgsm_metrics.accuracy_list[0],
                    'test_di-fgsm_auc': di_fgsm_metrics.auc_list[0],
                    'test_di-fgsm_asr': di_fgsm_metrics.asr_list[0],
                    'di-fgsm_correct_adv': di_fgsm_metrics.correct_adv_accuracy,
                    'di-fgsm_l2': di_fgsm_metrics.avg_l2,
                    'di-fgsm_linf': di_fgsm_metrics.avg_linf,
                    
                    'test_mi-fgsm_acc': mi_fgsm_metrics.accuracy_list[0],
                    'test_mi-fgsm_auc': mi_fgsm_metrics.auc_list[0],
                    'test_mi-fgsm_asr': mi_fgsm_metrics.asr_list[0],
                    'mi-fgsm_correct_adv': mi_fgsm_metrics.correct_adv_accuracy,
                    'mi-fgsm_l2': mi_fgsm_metrics.avg_l2,
                    'mi-fgsm_linf': mi_fgsm_metrics.avg_linf,
        
                    'test_pgd_acc': pgd_metrics.accuracy_list[0],
                    'test_pgd_auc': pgd_metrics.auc_list[0],
                    'test_pgd_asr': pgd_metrics.asr_list[0],
                    'pgd_correct_adv': pgd_metrics.correct_adv_accuracy,
                    'pgd_l2': pgd_metrics.avg_l2,
                    'pgd_linf': pgd_metrics.avg_linf,
                    'status': 'ok'})
        
            with open(path_name, 'w') as f:
                json.dump(results, f, indent=4)