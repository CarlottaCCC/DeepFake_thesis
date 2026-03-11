from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


#data = {
#    "Clean_Model": {
#        "No attack": {"accuracy": 0.75, "AUC score": 0.81},
#        "FGSM (eps=2/255)": {"accuracy": 0.52, "attack success rate": 0.30, "AUC score": 0.54},
#        "FGSM (eps=4/255)": {"accuracy": 0.35, "attack success rate": 0.53, "AUC score": 0.30},
#        "FGSM (eps=8/255)":  {"accuracy": 0.14, "attack success rate": 0.81, "AUC score": 0.08},
#        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.75, "attack success rate": 0.04, "AUC score": 0.81},
#    },
#     "FGSM_Model": {
#        "No attack": {"accuracy": 0.79, "AUC score": 0.87},
#        "FGSM (eps=2/255)": {"accuracy": 0.62, "attack success rate": 0.21, "AUC score": 0.66},
#        "FGSM (eps=4/255)": {"accuracy": 0.45, "attack success rate": 0.43, "AUC score": 0.42},
#        "FGSM (eps=8/255)": {"accuracy": 0.16, "attack success rate": 0.78, "AUC score": 0.09},
#        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.84, "attack success rate": 0.05, "AUC score": 0.92},
#    },
#     "SQUARE_Model": {
#        "No attack": {"accuracy": 0.76, "AUC score": 0.85},
#        "FGSM (eps=2/255)": {"accuracy": 0.56, "attack success rate": 0.26, "AUC score": 0.60},
#        "FGSM (eps=4/255)": {"accuracy": 0.42, "attack success rate": 0.44, "AUC score": 0.36},
#        "FGSM (eps=8/255)": {"accuracy": 0.20, "attack success rate": 0.74, "AUC score": 0.12},
#        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.78, "attack success rate": 0.04, "AUC score": 0.86},
#    },
#
#}

data = {
    "Clean_Model": {
        "No attack": {"accuracy": 0.89, "AUC score": 0.96},
        "FGSM (eps=2/255)": {"accuracy": 0.33, "attack success rate": 0.56, "AUC score": 0.23},
        "FGSM (eps=4/255)": {"accuracy": 0.35, "attack success rate": 0.53, "AUC score": 0.21},
        "FGSM (eps=8/255)":  {"accuracy": 0.14, "attack success rate": 0.81, "AUC score": 0.08},
        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.75, "attack success rate": 0.04, "AUC score": 0.81},
    },
     "FGSM-AT (epsilon=2/255)": {
        "No attack": {"accuracy": 0.62, "AUC score": 0.82},
        "FGSM (eps=2/255)": {"accuracy": 0.50, "attack success rate": 0.39, "AUC score": 0.27},
        "FGSM (eps=4/255)": {"accuracy": 0.30, "attack success rate": 0.59, "AUC score": 0.23},
        "FGSM (eps=8/255)": {"accuracy": 0.17, "attack success rate": 0.73, "AUC score": 0.07},
        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.84, "attack success rate": 0.05, "AUC score": 0.92},
    },
    "FGSM-AT (epsilon=4/255)": {
        "No attack": {"accuracy": 0.89, "AUC score": 0.96},
        "FGSM (eps=2/255)": {"accuracy": 0.39, "attack success rate": 0.22, "AUC score": 0.51},
        "FGSM (eps=4/255)": {"accuracy": 0.28, "attack success rate": 0.33, "AUC score": 0.13},
        "FGSM (eps=8/255)": {"accuracy": 0.14, "attack success rate": 0.52, "AUC score": 0.06},
        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.84, "attack success rate": 0.05, "AUC score": 0.92},
    },
    "FGSM-AT (epsilon=8/255)": {
        "No attack": {"accuracy": 0.89, "AUC score": 0.96},
        "FGSM (eps=2/255)": {"accuracy": 0.68, "attack success rate": 0.20, "AUC score": 0.74},
        "FGSM (eps=4/255)": {"accuracy": 0.54, "attack success rate": 0.31, "AUC score": 0.58},
        "FGSM (eps=8/255)": {"accuracy": 0.38, "attack success rate": 0.50, "AUC score": 0.34},
        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.84, "attack success rate": 0.05, "AUC score": 0.92},
    },
     "SQUARE_Model_1": {
        "No attack": {"accuracy": 0.89, "AUC score": 0.96},
        "FGSM (eps=2/255)": {"accuracy": 0.46, "attack success rate": 0.44, "AUC score": 0.40},
        "FGSM (eps=4/255)": {"accuracy": 0.24, "attack success rate": 0.66, "AUC score": 0.14},
        "FGSM (eps=8/255)": {"accuracy": 0.17, "attack success rate": 0.73, "AUC score": 0.07},
        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.78, "attack success rate": 0.04, "AUC score": 0.86},
    },
     "SQUARE_Model_2": {
        "No attack": {"accuracy": 0.84, "AUC score": 0.91},
        "FGSM (eps=2/255)": {"accuracy": 0.70, "attack success rate": 0.14, "AUC score": 0.73},
        "FGSM (eps=4/255)": {"accuracy": 0.49, "attack success rate": 0.36, "AUC score": 0.45},
        "FGSM (eps=8/255)": {"accuracy": 0.28, "attack success rate": 0.58, "AUC score": 0.17},
        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.78, "attack success rate": 0.04, "AUC score": 0.86},
    },

}

plot_model_metrics_by_attack(
    data=data,
    metrics=["accuracy", "attack success rate", "AUC score"],
    output_dir="comparison_images"
)