from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import foolbox as fb
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
import csv
from attacks import *
from attacks_implementation.nes import nes_attack, nes_attack_first_v
from attacks_implementation.jsma_accurate import tjsma_binary, wjsma_binary, wjsma_binary_debug
from attacks_implementation.autozoom import AutoZOOMAttack
from attacks_implementation.ifgsm import ifgsm_attack
from attacks_implementation.genattack import GenAttack


def test_attack(model, test_loader, attack_type, epsilon, library, model_name, model_data, attack_label_data, device, save_results=True):

    clean_metrics = Metrics()
    adv_metrics = Metrics()
    total_samples = 0

    model.eval()
    criterion = nn.CrossEntropyLoss()

    preprocessing = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    axis=-3  # per PyTorch (C, H, W)
    )
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing, device=device)

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

    #black box model wrapper for AutoZOOm attack
    model_fn = make_model_fn(model, device=device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    attack_fn = None

    if attack_type == 'square':
            attack_fn = SquareAttack(
            estimator=classifier,
            norm=np.inf,     
            eps=epsilon,
            max_iter=5000,
            p_init=0.8,
            nb_restarts=1,
            batch_size=32
            )

    elif attack_type == 'genattack':
        attack_fn = GenAttack(
            model=model, device=device, eps=epsilon,
            population_size=6, max_queries=5000,
            reduced_dim=56,          # 4x reduction on each spatial dim, matches paper's ratio choice
            upsample_mode="nearest", # paper's default
        )
    print(f"Starting {attack_type} test!")
    pbar = tqdm(test_loader, desc=f"Testing", unit="batch")
    for batch in pbar:
        if batch is None:
            continue
        imgs, labels, _ = batch
        imgs, labels = imgs.to(device), labels.to(device).long().squeeze()
        batch_size = imgs.size(0)
        #print(y.shape, y.dtype)
        norm_imgs = normalize(imgs)

        with torch.no_grad():
            clean_preds = model(norm_imgs).argmax(1)
        clean_correct_mask = (clean_preds == labels)
    
        # Only attack examples the model already gets right
        imgs_correct = imgs[clean_correct_mask]
        labels_correct = labels[clean_correct_mask]

        imgs_incorrect = imgs[~clean_correct_mask]
        labels_incorrect = labels[~clean_correct_mask]

        # ATTACKS
        #if attack_type == 'fgsm' and library == 'None':
        #    grad_imgs = torch.autograd.grad(loss_clean, imgs, retain_graph=True, create_graph=False)[0]
        #    imgs_adv = imgs +  epsilon * grad_imgs.sign()
        #    #imgs_adv = torch.clamp(imgs_adv, 0, 1)
        #    imgs_adv = torch.clamp(imgs_adv, imgs - epsilon, imgs + epsilon)

        if attack_type == 'rs_fgsm' and library == 'None':
            imgs_adv = rs_fgsm(epsilon, model, imgs, labels)

        if attack_type == 'pgd' and library == 'None':
            imgs_adv = standard_pgd_attack(model, imgs.detach(), labels, epsilon, alpha=2/255, steps=20, normalize=normalize)
            imgs_adv = normalize(imgs_adv.detach())
        if library == 'foolbox' and attack_type != 'genattack':
            attack_fn = get_attack_foolbox(attack_type)
            eps = epsilon
            _, imgs_adv, _ = attack_fn(fmodel, imgs, labels, epsilons=eps)
            imgs_adv = normalize(imgs_adv.detach())
        elif library == 'art' and (attack_type!= 'jsma'):
            attack_fn = get_attack_art(attack_type, classifier)
            print(classifier._device)
            #target_labels = 1 - labels
            imgs_adv = attack_fn.generate(x=imgs.cpu().numpy(), y=labels.cpu().numpy())
            # ART returns numpy, need to convert imgs_square to tensor to pass it to the model
            imgs_adv = torch.from_numpy(imgs_adv).float().to(device)
            imgs_adv = normalize(imgs_adv.detach())
        elif library == 'art' and (attack_type == 'jsma'):
            attack_fn = get_attack_art(attack_type, classifier)
            target_labels = 1 - labels
            imgs_adv = attack_fn.generate(x=imgs.cpu().numpy(), y=target_labels.cpu().numpy())
            # ART returns numpy, need to convert imgs_square to tensor to pass it to the model
            imgs_adv = normalize(imgs_adv.detach())
            imgs_adv = torch.from_numpy(imgs_adv).float().to(device)

        elif attack_type == 'nes' and library == 'None':
            imgs_adv = nes_attack_first_v(model, imgs_correct, labels_correct, eps=epsilon, 
                                  sigma=0.001, n_samples=50, step_size=2/255, 
                                  n_iters=100, device=device)

            imgs_adv = normalize(imgs_adv.detach())
        elif attack_type == 'square' and library == 'None':
             imgs_adv = attack_fn.generate(x=imgs_correct.cpu().numpy(), y=labels_correct.cpu().numpy())
             # ART returns numpy, need to convert imgs_square to tensor to pass it to the model
             imgs_adv = torch.from_numpy(imgs_adv).float().to(device)
             imgs_adv = normalize(imgs_adv.detach())
        elif attack_type == 'genattack' and library == 'None':
             imgs_adv, success, n_queries = attack_fn.attack_batch(imgs_correct,labels_correct)
             imgs_adv = normalize(imgs_adv.detach())
             
        #elif library == 'foolbox' and attack_type == 'genattack':
        #    # CHIAVE: Crea target labels (classe opposta per classificazione binaria)
        #    target_labels = 1 - labels  # Se binario: 0→1, 1→0
        #    # Crea criterion TARGETED 
        #    criterion = fb.criteria.TargetedMisclassification(target_labels)
        #    attack_fn = get_attack_foolbox(attack_type)
        #    eps = epsilon
        #    _, imgs_adv, _ = attack_fn(fmodel, imgs, criterion, epsilons=eps)
        #    imgs_adv = normalize(imgs_adv.detach())

        #normalize
        imgs = normalize(imgs_correct.detach())

        # compute L2 and Linf metrics
        #delta = imgs_adv - imgs
        delta = imgs_adv - imgs_correct
        l2, linf = batch_norms(delta)
        adv_metrics.total_l2 += l2.sum().item()
        adv_metrics.total_linf += linf.sum().item()

        # normalize
        #imgs_adv = normalize(imgs_adv.detach())
        #print(f"min: {imgs.min():.3f}, max: {imgs.max():.3f}")
        #print(f"min: {imgs_adv.min():.3f}, max: {imgs_adv.max():.3f}")

        # inferenza
        with torch.no_grad():
            logits_clean = model(imgs)
            logits_adv = model(imgs_adv)

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            #debug
            preds_clean = torch.argmax(logits_clean, dim=1)
            preds_adv = torch.argmax(logits_adv, dim=1)
            clean_metrics.update(labels_correct, probs_clean)
            probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
            adv_metrics.update(labels_correct, probs_adv)

        #correct_adv += (preds_adv == y_to_attack).sum().item()

        total_samples += batch_size
    
    #Attack success rate
    adv_metrics.attack_success_rate(clean_metrics.all_probs)
    # Average L2 and L_inf norm
    adv_metrics.avg_l2 = adv_metrics.total_l2/total_samples
    adv_metrics.avg_linf = adv_metrics.total_linf/total_samples
   
    clean_results = clean_metrics.compute()
    adv_results = adv_metrics.compute()
    #adv_metrics.correct_adv_accuracy = correct_adv
    
    print("CLEAN RESULTS")
    clean_metrics.print(0)
    print(f"{attack_type} ATTACK RESULTS")
    adv_metrics.print(0)
    print(f"Attack Success Rate:  {adv_metrics.asr_list[0]}")
    print(f"Average L2:  {adv_metrics.avg_l2}")
    print(f"Average Linf:  {adv_metrics.avg_linf}")
   
    if save_results==True:

        #saving metrics history
        history = {
            "number_of_samples": total_samples,
            "clean_auc": clean_metrics.auc_list,
            f"{attack_type}_auc": adv_metrics.auc_list,
            "clean_auc": clean_metrics.auc_list,
            f"{attack_type}_auc": adv_metrics.auc_list,
            "clean_f1": clean_metrics.f1_list,
            f"{attack_type}_f1": adv_metrics.f1_list,
            "clean_precision": clean_metrics.precision_list,
            f"{attack_type}_precision": adv_metrics.precision_list,
            "clean_recall": clean_metrics.recall_list,
            f"{attack_type}_recall": adv_metrics.recall_list,
            "clean_accuracy": clean_metrics.accuracy_list,
            f"{attack_type}_accuracy": adv_metrics.accuracy_list,
            f"{attack_type}_asr": adv_metrics.asr_list,
            "fgsm_epsilon_train": EPS,
            f"{attack_type}_epsilon_test": epsilon,
            f"{attack_type}_avg_l2": adv_metrics.avg_l2,
            f"{attack_type}_avg_linf": adv_metrics.avg_linf
        }
        save_history_json(history,f"test_results/prova_test/{model_name}/test_results_{attack_type}/results_{attack_type}_eps_{epsilon}_num_samples_{total_samples}_norm.json")
    
        save_results_for_data(
            model_name=model_data,
            attack_label=attack_label_data,
            accuracy=adv_metrics.accuracy_list[0],
            auc=adv_metrics.auc_list[0],
            asr=adv_metrics.asr_list[0],
            avg_l2=adv_metrics.avg_l2,
            avg_linf=adv_metrics.avg_linf,
            output_path=f"test_results_for_data/prova_test/{model_name}/test_results_{attack_type}/results_{attack_type}_eps_{epsilon}_num_samples_{total_samples}_norm.json"
        )

    return clean_metrics, adv_metrics


