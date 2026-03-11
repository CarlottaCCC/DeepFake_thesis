from torchvision.models import resnet50
from torch.utils.data import DataLoader
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from test_jsma import test_jsma
import foolbox as fb
import torchattacks
import torch.optim as optim
import csv


def test_white(model, test_loader, model_name, device):

    clean_metrics = Metrics()
    fgsm_metrics = Metrics()
    pgd_metrics = Metrics()
    ifgsm_metrics = Metrics()
    gen_metrics = Metrics()
    total_samples = 0

    # I define the image bounds for the fmodel in order to properly attack in that space
    #mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    #std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
#
    #lower = (0 - mean) / std
    #upper = (1 - mean) / std
    model.eval()

    #fmodel = fb.PyTorchModel(model, bounds=(lower.min().item(), upper.max().item()), device=device)
    fmodel = fb.PyTorchModel(model, bounds=(0,1), device=device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    

    # ATTACKS *****************************************
    # GEN ATTACK
    # GenAttack requires a specific target
    gen_attack = fb.attacks.GenAttack(steps=1000)

    # FGSM
    fgsm = fb.attacks.FGSM()

    # PGD
    pgd = fb.attacks.LinfPGD(
    steps=20,
    rel_stepsize=2/40,
    abs_stepsize=None,
    random_start=True)

    #IFGSM
    ifgsm = fb.attacks.LinfPGD(
    steps=10,
    rel_stepsize=2/10,
    random_start=False)  # → with no random start PGD = IFGSM

    pbar = tqdm(test_loader, desc=f"Testing", unit="batch")

    for batch in pbar:
        if batch is None:
            continue
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device).long().squeeze()
        #print(y.shape, y.dtype)
        batch_size = imgs.size(0)

        # FGSM
        print("Performing FGSM attack")
        _, imgs_fgsm, _ = fgsm(fmodel, imgs, labels, epsilons=TEST_EPS_FGSM)

        delta = imgs_fgsm - imgs
        l2, linf = batch_norms(delta)
        fgsm_metrics.total_l2 += l2.sum().item()
        fgsm_metrics.total_linf += linf.sum().item()
        
        # PGD
        print("Performing PGD attack")
        _, imgs_pgd, _ = pgd(fmodel, imgs, labels, epsilons=EPS_PGD)

        delta = imgs_pgd - imgs
        l2, linf = batch_norms(delta)
        pgd_metrics.total_l2 += l2.sum().item()
        pgd_metrics.total_linf += linf.sum().item()

        # IFGSM
        print("Performing IFGSM attack")
        _, imgs_ifgsm, _ = ifgsm(fmodel, imgs, labels, epsilons=EPS_IFGSM)

        delta = imgs_ifgsm - imgs
        l2, linf = batch_norms(delta)
        ifgsm_metrics.total_l2 += l2.sum().item()
        ifgsm_metrics.total_linf += linf.sum().item()

        # GEN ATTACK
        # target labels (opposite class)
        target_labels = 1 - labels  # (0→1, 1→0)
        # Usa TargetedMisclassification criterion
        criterion_gen = fb.criteria.TargetedMisclassification(target_labels)
        print("Performing GEN attack")
        _, imgs_gen, _ = gen_attack(fmodel, imgs, criterion_gen, epsilons=EPS_IFGSM)

        delta = imgs_gen - imgs
        l2, linf = batch_norms(delta)
        gen_metrics.total_l2 += l2.sum().item()
        gen_metrics.total_linf += linf.sum().item()

        # Normalization
        imgs = normalize(imgs)
        imgs_fgsm = normalize(imgs_fgsm)
        imgs_pgd = normalize(imgs_pgd)
        imgs_ifgsm = normalize(imgs_ifgsm)

        # inferenza
        with torch.no_grad():
            logits_clean = model(imgs)
            logits_fgsm = model(imgs_fgsm)
            logits_pgd = model(imgs_pgd)
            logits_ifgsm = model(imgs_ifgsm)
            logits_gen = model(imgs_gen)

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            clean_metrics.update(labels, probs_clean)
            probs_fgsm = torch.softmax(logits_fgsm, dim=1)[:,1].detach().cpu().numpy()
            fgsm_metrics.update(labels, probs_fgsm)
            probs_pgd = torch.softmax(logits_pgd, dim=1)[:,1].detach().cpu().numpy()
            pgd_metrics.update(labels, probs_pgd)
            probs_ifgsm = torch.softmax(logits_ifgsm, dim=1)[:,1].detach().cpu().numpy()
            ifgsm_metrics.update(labels, probs_ifgsm)
            probs_gen = torch.softmax(logits_gen, dim=1)[:,1].detach().cpu().numpy()
            gen_metrics.update(labels, probs_gen)

        total_samples += batch_size
    
    #Attack success rate
    fgsm_metrics.attack_success_rate(clean_metrics.all_probs)
    pgd_metrics.attack_success_rate(clean_metrics.all_probs)
    ifgsm_metrics.attack_success_rate(clean_metrics.all_probs)
    gen_metrics.attack_success_rate(clean_metrics.all_probs)

    # Average L2 and L_inf norm
    fgsm_metrics.avg_l2 = fgsm_metrics.total_l2/total_samples
    fgsm_metrics.avg_linf = fgsm_metrics.total_linf/total_samples

    pgd_metrics.avg_l2 = pgd_metrics.total_l2/total_samples
    pgd_metrics.avg_linf = pgd_metrics.total_linf/total_samples

    ifgsm_metrics.avg_l2 = ifgsm_metrics.total_l2/total_samples
    ifgsm_metrics.avg_linf = ifgsm_metrics.total_linf/total_samples

    gen_metrics.avg_l2 = gen_metrics.total_l2/total_samples
    gen_metrics.avg_linf = gen_metrics.total_linf/total_samples

    
    clean_results = clean_metrics.compute()
    fgsm_results = fgsm_metrics.compute()
    pgd_results = pgd_metrics.compute()
    ifgsm_results = ifgsm_metrics.compute()
    gen_results = gen_metrics.compute()
    
    print("CLEAN RESULTS")
    clean_metrics.print(0)
    print("FGSM ATTACK RESULTS (white-box)")
    fgsm_metrics.print(0)
    print(f"Attack Success Rate:  {fgsm_metrics.asr_list[0]}")
    print(f"Average L2 norm: {fgsm_metrics.avg_l2}")
    print(f"Average L_inf norm: {fgsm_metrics.avg_linf}")

    print("PGD ATTACK RESULTS (white-box)")
    pgd_metrics.print(0)
    print(f"Attack Success Rate:  {pgd_metrics.asr_list[0]}")
    print(f"Average L2 norm: {pgd_metrics.avg_l2}")
    print(f"Average L_inf norm: {pgd_metrics.avg_linf}")

    print("IFGSM ATTACK RESULTS (white-box)")
    ifgsm_metrics.print(0)
    print(f"Attack Success Rate:  {ifgsm_metrics.asr_list[0]}")
    print(f"Average L2 norm: {ifgsm_metrics.avg_l2}")
    print(f"Average L_inf norm: {ifgsm_metrics.avg_linf}")

    print("GEN ATTACK RESULTS (white-box)")
    gen_metrics.print(0)
    print(f"Attack Success Rate:  {gen_metrics.asr_list[0]}")
    print(f"Average L2 norm: {gen_metrics.avg_l2}")
    print(f"Average L_inf norm: {gen_metrics.avg_linf}")

    #saving metrics history
    history = {
        "clean_auc": clean_metrics.auc_list,
        "clean_auc": clean_metrics.auc_list,
        "clean_f1": clean_metrics.f1_list,
        "clean_precision": clean_metrics.precision_list,
        "clean_recall": clean_metrics.recall_list,
        "clean_accuracy": clean_metrics.accuracy_list,
        "fgsm_auc": fgsm_metrics.auc_list,
        "fgsm_auc": fgsm_metrics.auc_list,
        "fgsm_f1": fgsm_metrics.f1_list,
        "fgsm_precision": fgsm_metrics.precision_list,
        "fgsm_recall": fgsm_metrics.recall_list,
        "fgsm_accuracy": fgsm_metrics.accuracy_list,
        "fgsm_asr": fgsm_metrics.asr_list,
        "fgsm_avg_l2": fgsm_metrics.avg_l2,
        "fgsm_avg_linf": fgsm_metrics.avg_linf,
        "fgsm_epsilon_train": EPS,
        "fgsm_epsilon_test": TEST_EPS_FGSM,
        "pgd_auc": pgd_metrics.auc_list,
        "pgd_f1": pgd_metrics.f1_list,
        "pgd_precision": pgd_metrics.precision_list,
        "pgd_recall": pgd_metrics.recall_list,
        "pgd_accuracy": pgd_metrics.accuracy_list,
        "pgd_asr": pgd_metrics.asr_list,
        "pgd_avg_l2": pgd_metrics.avg_l2,
        "pgd_avg_linf": pgd_metrics.avg_linf,
        "pgd_epsilon_test": EPS_PGD,
        "ifgsm_auc": ifgsm_metrics.auc_list,
        "ifgsm_f1": ifgsm_metrics.f1_list,
        "ifgsm_precision": ifgsm_metrics.precision_list,
        "ifgsm_recall": ifgsm_metrics.recall_list,
        "ifgsm_accuracy": ifgsm_metrics.accuracy_list,
        "ifgsm_asr": ifgsm_metrics.asr_list,
        "ifgsm_avg_l2": ifgsm_metrics.avg_l2,
        "ifgsm_avg_linf": ifgsm_metrics.avg_linf,
        "ifgsm_epsilon_test": EPS_IFGSM,
        "gen_auc": gen_metrics.auc_list,
        "gen_f1": gen_metrics.f1_list,
        "gen_precision": gen_metrics.precision_list,
        "gen_recall": gen_metrics.recall_list,
        "gen_accuracy": gen_metrics.accuracy_list,
        "gen_asr": gen_metrics.asr_list,
        "gen_avg_l2": gen_metrics.avg_l2,
        "gen_avg_linf": gen_metrics.avg_linf
    }
    save_history_json(history,f"test_results_white/{model_name}/results_white_eps_{TEST_EPS_FGSM}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}.json")

    return clean_metrics, fgsm_metrics, pgd_metrics, ifgsm_metrics

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=None)
    # I modify the last layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # I load the trained clean model
    model_name = "resnet50_clean_epoch_40_LR_0.0003_batchsize_32_WD_1e-05"
    checkpoint_path = f"models_10/clean_resnet50/{model_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

    transform_jsma = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    ])
  
    print("Initializing testing dataset....")
    test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
    # I get a smaller subset of 500 images
    test_small = balanced_subset(test_dataset, n_per_class=50)
    
    print("Initializing test loader...")
    test_loader = DataLoader(test_small, batch_size=BATCH_SIZE, shuffle=True)

    clean_metrics, fgsm_metrics, pgd_metrics, ifgsm_metrics = test_white(model, test_loader, model_name, device)

    # JSMA TEST -> NEEDS SUBSAMPLED IMAGES
    #print("Initializing testing dataset for JSMA....")
    #test_dataset_jsma = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform_jsma)
    ## I get a smaller subset of 500 images
    #test_small_jsma = balanced_subset(test_dataset_jsma, n_per_class=50)
    #
    #print("Initializing test loader...")
    #test_loader_jsma = DataLoader(test_small_jsma, batch_size=BATCH_SIZE, shuffle=True)
    #print("Staring JSMA test.........")
    #jsma_metrics = test_jsma(model, test_loader_jsma, device)

    #root = "metrics_results_white"
    ## plot roc curve
    #plot_roc(clean_metrics.fpr, clean_metrics.tpr, clean_metrics.auc_list[0], "(test)", f"{root}/clean_test")
    #plot_roc(fgsm_metrics.fpr, fgsm_metrics.tpr, fgsm_metrics.auc_list[0], "(test)", f"{root}/fgsm_test")
    #plot_roc(pgd_metrics.fpr, pgd_metrics.tpr, pgd_metrics.auc_list[0], "(test)", f"{root}/pgd_test")
    #plot_roc(ifgsm_metrics.fpr, ifgsm_metrics.tpr, ifgsm_metrics.auc_list[0], "(test)", f"{root}/ifgsm_test")
    #plot_roc(jsma_metrics.fpr, jsma_metrics.tpr, jsma_metrics.auc_list[0], "(test)", f"{root}/jsma_test")