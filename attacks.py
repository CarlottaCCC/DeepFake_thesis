from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import json
import os
import torch
import random
from torch.utils.data import Subset
from torchvision import transforms
from utils import *
import foolbox as fb
from art.attacks.evasion import (
    SaliencyMapMethod,      # JSMA
    SquareAttack,           # Square Attack
    ZooAttack              # ZOO
)
from attacks_implementation.nes_2 import *
from attacks_implementation.autozoom_bilin import *

# No NES in ART
# No autoZOOM in ART
# and no Gen attack

# ********* WHITE BOX ATTACKS *********
# PGD
EPS_PGD = 8/255
#IFGSM
EPS_IFGSM = 8/255

transform_size = transforms.Compose([
transforms.Resize((224, 224))
])

transform_jsma = transforms.Compose([
transforms.Resize((64, 64))
])

# Normalization - I needto normalize the images after they are attacked
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

# Define the fmodel in order to use the foolbox attacks
def get_fmodel(model, device):
    fmodel = fb.PyTorchModel(model, bounds=(0,1), device=device)
    return fmodel

def make_model_fn(model: nn.Module, device: str = 'cpu'):
    model = model.to(device).eval()

    def model_fn(images_np: np.ndarray) -> np.ndarray:
        t = torch.tensor(images_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(t)
        return logits.cpu().numpy()

    return model_fn

# Function that returns a chosen attack
#def get_images_by_attack(attack_type, model, imgs, labels):
#    if attack_type == 'fgsm':
#        _, imgs_fgsm, _ = fgsm(model, imgs, labels, epsilons=TEST_EPS_FGSM)
#        return normalize(imgs_fgsm)
#    elif attack_type == 'pgd_weak':
#        _, imgs_pgd, _ = pgd(model, imgs, labels, eps=0.02, steps=5)
#        return normalize(imgs_pgd)
#    elif attack_type == 'pgd_strong':
#        _, imgs_pgd, _ = pgd(model, imgs, labels, eps=0.03, steps=20)
#        return normalize(imgs_pgd)
#    elif attack_type == 'ifgsm':
#        _, imgs_ifgsm, _ = ifgsm(model, imgs, labels, epsilons=EPS_IFGSM)
#        return normalize(imgs_ifgsm)

def rs_fgsm(current_eps, model, imgs_raw, y):
    with torch.enable_grad():
        imgs_raw = imgs_raw.detach()
        delta = torch.empty_like(imgs_raw).uniform_(-current_eps, current_eps)
        delta = torch.clamp(imgs_raw + delta, 0, 1) - imgs_raw
        imgs_start = (imgs_raw + delta).clone().requires_grad_(True)
        imgs_norm = normalize(imgs_start)
        logits_clean = model(imgs_norm)
        loss_tmp = nn.CrossEntropyLoss()(logits_clean, y)
        grad_imgs = torch.autograd.grad(loss_tmp, imgs_start)[0]
 
    imgs_adv = imgs_start.detach() + current_eps * grad_imgs.sign()
    imgs_adv = torch.clamp(imgs_adv, 0, 1)

    return imgs_adv
    
def get_attack_foolbox(attack_type):
    if attack_type == 'fgsm':
       fgsm = fb.attacks.FGSM()
       return fgsm
    elif attack_type == 'pgd':
        pgd = fb.attacks.LinfPGD(
        steps=20,
        #rel_stepsize=1/255,
        abs_stepsize=1/255,
        random_start=True)
        return pgd
    elif attack_type == 'ifgsm':
        ifgsm = fb.attacks.LinfPGD(
        steps=20,
        abs_stepsize=1/255,
        random_start=False)  # → with no random start PGD = IFGSM
        return ifgsm
    elif attack_type == 'genattack':
        genattack = fb.attacks.GenAttack()
        return genattack
        
       
    
def get_attack_art(attack_type, classifier):
    if attack_type == 'square':
        square = SquareAttack(
        estimator=classifier,
        norm="inf",     
        eps=16/255,
        max_iter=SQUARE_ITER,
        p_init=0.8
        )
        return square
    elif attack_type == 'zoo': # ZOO is very slow, needs less images to be tested on
        zoo = ZooAttack(
            classifier=classifier,
            targeted=False,        # False = untargeted
            max_iter=200,          # Numero massimo di iterazioni
            confidence=0.0,        # Parametro di ottimizzazione
            learning_rate=0.01,    # Step size
            binary_search_steps=5, # Numero di ricerche per trovare il miglior perturbation
            initial_const=1e-1,    # Costante iniziale
            abort_early=True,
            use_resize=True
        )
        return zoo
    elif attack_type == 'jsma':
        jsma = SaliencyMapMethod(classifier=classifier, batch_size=64)
        return jsma

    elif attack_type == 'autozoom':
        autozoom = AutoZoomBilin(
        estimator=classifier,
        max_iter=500,
        learning_rate=1e-2,
        binary_search_steps=5,
        init_const=1.0,
        confidence=0.0,
        targeted=False,         # True se vuoi un attacco mirato
        reduce_factor=4,        # 224→56 nello spazio ridotto (~16x meno coord.)
        num_random_vecs=1,      # 1 per efficienza; aumenta per stime più accurate
        h=1e-4,
        clip_min=0.0,
        clip_max=1.0,
        verbose=100,
    )
        return autozoom

def get_attack(attack_type, model):
    if attack_type == "nes":
        nes = NES(
        model=model,
        device='cuda',
        norm=np.inf,
        eps=4/255,
        stepsize=1/255,
        nes_samples=20,
        sample_per_draw=20,
        max_queries=1000,
        search_sigma=0.02,
        target=True
    )
    return nes
 


