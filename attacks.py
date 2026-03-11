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
    
def get_attack_foolbox(attack_type):
    if attack_type == 'fgsm':
       fgsm = fb.attacks.FGSM()
       return fgsm
    elif attack_type == 'pgd':
        pgd = fb.attacks.LinfPGD(
        steps=20,
        rel_stepsize=2/8,
        abs_stepsize=None,
        random_start=True)
        return pgd
    elif attack_type == 'ifgsm':
        ifgsm = fb.attacks.LinfPGD(
        steps=20,
        rel_stepsize=2/8,
        random_start=False)  # → with no random start PGD = IFGSM
        return ifgsm
    elif attack_type == 'genattack':
        genattack = fb.attacks.GenAttack()
        return genattack
    elif attack_type == 'hopskipjump':
        hopskipjump = fb.attacks.HopSkipJumpAttack()
        return hopskipjump
        
       
    
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
            targeted=True,        # False = untargeted
            max_iter=50,          # Numero massimo di iterazioni
            confidence=0.0,        # Parametro di ottimizzazione
            learning_rate=0.01,    # Step size
            binary_search_steps=1, # Numero di ricerche per trovare il miglior perturbation
            initial_const=1e-2,    # Costante iniziale
            abort_early=True,
        )
        return zoo


