import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
from torchvision import transforms
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from utils import *
import itertools
from test_attacks import test_attack

def test_white_box(path_name, mode_log, model, test_loader, device):

    results = []

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
        'mode_log': mode_log,
        'status': 'ok'})

    out_dir = "test_white_box_results"
    os.makedirs(out_dir, exist_ok=True)

    with open(path_name, 'w') as f:
        json.dump(results, f, indent=4)

def test_black_box(model_label, model, test_loader, epsilon, device):

    results_nes = []
    results_gen = []
    results_square = []

    # NES
    clean_metrics, nes_metrics_1 = test_attack(model, test_loader, 'nes', epsilon, 'None', " ", " ", "", device, save_results=False)

    results_nes.append({
            'test_clean_acc': clean_metrics.accuracy_list[0],
            'test_clean_auc': clean_metrics.auc_list[0],
            'test_clean_asr': 0,
            'test_nes_acc': nes_metrics_1.accuracy_list[0],
            'test_nes_auc': nes_metrics_1.auc_list[0],
            'test_nes_asr': nes_metrics_1.asr_list[0],
            'status': 'ok'})

    # save nes results
    out_dir = f"test_black_box_results/{model_label}"
    os.makedirs(out_dir, exist_ok=True)
    path_name_nes = f"test_black_box_results/{model_label}/{model_label}_nes_results_eps_{epsilon*255}_2.json"

    with open(path_name_nes, 'w') as f:
        json.dump(results_nes, f, indent=4)


    # GENATTACK
    clean_metrics, gen_metrics_1 = test_attack(model, test_loader, 'genattack', epsilon, 'None', " ", " ", "", device, save_results=False)

    results_gen.append({
           'test_clean_acc': clean_metrics.accuracy_list[0],
           'test_clean_auc': clean_metrics.auc_list[0],
           'test_clean_asr': 0,
           'test_gen_acc': gen_metrics_1.accuracy_list[0],
           'test_gen_auc': gen_metrics_1.auc_list[0],
           'test_gen_asr': gen_metrics_1.asr_list[0],
           'status': 'ok'})

    # save genattack results
    os.makedirs(out_dir, exist_ok=True)
    path_name_gen = f"test_black_box_results/{model_label}/{model_label}_gen_results_eps_{epsilon*255}_2.json"

    with open(path_name_gen, 'w') as f:
        json.dump(results_gen, f, indent=4)

    # SQUARE
    clean_metrics, square_metrics_1 = test_attack(model, test_loader, 'square', epsilon, 'None', " ", " ", "", device, save_results=False)
    #clean_metrics, square_metrics_2 = test_attack(model, test_loader, 'square', 16/255, 'None', " ", " ", "", device, save_results=False)

    results_square.append({
            'test_clean_acc': clean_metrics.accuracy_list[0],
            'test_clean_auc': clean_metrics.auc_list[0],
            'test_clean_asr': 0,
            'test_square_acc': square_metrics_1.accuracy_list[0],
            'test_square_auc': square_metrics_1.auc_list[0],
            'test_square_asr': square_metrics_1.asr_list[0],
            'status': 'ok'})

    # save SQUARE results
    os.makedirs(out_dir, exist_ok=True)
    path_name_square = f"test_black_box_results/{model_label}/{model_label}_square_results_eps_{epsilon*255}_2.json"

    with open(path_name_square, 'w') as f:
        json.dump(results_square, f, indent=4)

if __name__ == "__main__":
    seed=42
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)
    #model = model.cuda()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader, val_loader, test_loader = get_data_loaders(transform, 32)
    #models_names = [{"model_label":"PGD-AT linear eps sched", "model_name":"pgdat_ades_difat/resnet50_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25_freeze.pt"},
    #                        {"model_label":"ADES", "model_name":"pgdat_ades_difat/resnet50_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_50__lr_0.001_seed_42_epochs_25_freeze_norampup.pt"},
    #                        {"model_label":"ADES + 11 epochs PGD-AT", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_ades_baseline_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
    #                        {"model_label":"PGD-AT + DiFat conf 2", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_7__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
    #                        {"model_label":"PGD-AT + DiFat conf 4", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_[3, 6, 13]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
    #                        {"model_label":"ADES + PGD-AT + DiFat", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_ades_baseline_difat_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23]_epochsdifat_[24]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
    #                        {"model_label":"baseline model", "model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"}]

    models_names = [{"model_label":"PGD-AT + DiFat conf 2", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_7__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"PGD-AT + DiFat conf 4", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_[3, 6, 13]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"ADES + PGD-AT + DiFat", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_ades_baseline_difat_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23]_epochsdifat_[24]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"baseline model", "model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"}]

    

    for model_item in models_names:

        model_label = model_item["model_label"]
        model_name = model_item["model_name"]
        checkpoint_path = f"{MODELS_DIR}/{model_name}"
        model = reset_model(checkpoint_path,device)

        test_black_box(model_label, model, test_loader, epsilon=16/255, device=device)
    