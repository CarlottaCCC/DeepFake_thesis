from torchvision.models import resnet50
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
import csv
import torchattacks

# function that divides an image in patches such that they are not overlapping
# returns:
# patches: tensor [num_patches, C, patch_size, patch_size]
#patch_labels: tensor [B*num_patches] con label replicate
#batch_indices: tensor [B*num_patches] indica da quale immagine viene ogni patch
#positions: lista di tuple (i, j) posizioni delle patches
# positions: tuples list (i,j) with the patches positions
def divide_into_patches_batch(images, labels, patch_size=16):
    B, C, H, W = images.shape
    
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image size {H}x{W} must be divisible by patch_size {patch_size}"
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Dividi in patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, num_patches, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)
    # Shape: [B, num_patches, C, patch_size, patch_size]
    
    # Flatten: [B*num_patches, C, patch_size, patch_size]
    patches_flat = patches.reshape(-1, C, patch_size, patch_size)
    
    # Replica le label per ogni patch dell'immagine
    # Se immagine 0 ha label 1, tutte le sue 196 patches avranno label 1
    patch_labels = labels.unsqueeze(1).repeat(1, num_patches).reshape(-1)
    # Shape: [B*num_patches]
    
    # Tiene traccia da quale immagine del batch originale viene ogni patch
    batch_indices = torch.arange(B, device=images.device).unsqueeze(1).repeat(1, num_patches).reshape(-1)
    # Shape: [B*num_patches]
    
    # Posizioni
    positions = [(i, j) for i in range(num_patches_h) for j in range(num_patches_w)]
    
    return patches_flat, patch_labels, batch_indices, positions


def test_jsma(model, test_loader, device):

    jsma_metrics = Metrics()
    clean_metrics = Metrics()

    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    pbar = tqdm(test_loader, desc=f"Testing", unit="batch")

    for batch in pbar:
        if batch is None:
            continue
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device).long().squeeze()
        #print(y.shape, y.dtype)

        # divide the images in patches 
        patches, patches_labels, _, _ = divide_into_patches_batch(imgs, labels)

        #JSMA
        jsma = torchattacks.JSMA(model, theta=1.0, gamma=0.1)
        imgs_jsma = jsma(patches, patches_labels)

        imgs = normalize(imgs)
        imgs_jsma = normalize(imgs_jsma)

        # inferenza
        with torch.no_grad():
            logits_clean = model(imgs)
            logits_jsma = model(imgs_jsma)

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            clean_metrics.update(labels, probs_clean)
            probs_jsma = torch.softmax(logits_jsma, dim=1)[:,1].detach().cpu().numpy()
            jsma_metrics.update(labels, probs_jsma)
    
    #Attack success rate
    jsma_metrics.attack_success_rate(clean_metrics.all_probs)
   
    clean_results = clean_metrics.compute()
    jsma_results = jsma_metrics.compute()
    
    print("RESULTS JSMA TESTING -> SUBSAMPLED IMAGES")
    print("CLEAN RESULTS")
    clean_metrics.print(0)
    print("JSMA ATTACK RESULTS (white-box)")
    jsma_metrics.print(0)
    print(f"Attack Success Rate:  {jsma_metrics.asr_list[0]}")
   

    #saving metrics history
    #history = {
    #    "clean_auc": clean_metrics.auc_list,
    #    "jsma_auc": jsma_metrics.auc_list,
    #    "clean_auc": clean_metrics.auc_list,
    #    "jsma_auc": jsma_metrics.auc_list,
    #    "clean_f1": clean_metrics.f1_list,
    #    "jsma_f1": jsma_metrics.f1_list,
    #    "clean_precision": clean_metrics.precision_list,
    #    "jsma_precision": jsma_metrics.precision_list,
    #    "clean_recall": clean_metrics.recall_list,
    #    "jsma_recall": jsma_metrics.recall_list,
    #    "clean_accuracy": clean_metrics.accuracy_list,
    #    "jsma_accuracy": jsma_metrics.accuracy_list,
    #    "jsma_asr": jsma_metrics.asr_list,
    #    "fgsm_epsilon_train": EPS
    #}
    #save_history_json(history,f"test_results_jsma/trained_square_results/results_square_eps_{TEST_EPS_FGSM}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}.json")

    return jsma_metrics