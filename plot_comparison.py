from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


data = {
    "Clean_Model": {
        "No attack": {"accuracy": 0.75, "AUC score": 0.81},
        "FGSM (eps=2/255)": {"accuracy": 0.52, "attack success rate": 0.30, "AUC score": 0.54},
        "FGSM (eps=4/255)": {"accuracy": 0.35, "attack success rate": 0.53, "AUC score": 0.30},
        "FGSM (eps=8/255)":  {"accuracy": 0.14, "attack success rate": 0.81, "AUC score": 0.08},
        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.75, "attack success rate": 0.04, "AUC score": 0.81},
    },
     "FGSM_Model": {
        "No attack": {"accuracy": 0.79, "AUC score": 0.87},
        "FGSM (eps=2/255)": {"accuracy": 0.62, "attack success rate": 0.21, "AUC score": 0.66},
        "FGSM (eps=4/255)": {"accuracy": 0.45, "attack success rate": 0.43, "AUC score": 0.42},
        "FGSM (eps=8/255)": {"accuracy": 0.16, "attack success rate": 0.78, "AUC score": 0.09},
        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.84, "attack success rate": 0.05, "AUC score": 0.92},
    },
     "SQUARE_Model": {
        "No attack": {"accuracy": 0.76, "AUC score": 0.85},
        "FGSM (eps=2/255)": {"accuracy": 0.56, "attack success rate": 0.26, "AUC score": 0.60},
        "FGSM (eps=4/255)": {"accuracy": 0.42, "attack success rate": 0.44, "AUC score": 0.36},
        "FGSM (eps=8/255)": {"accuracy": 0.20, "attack success rate": 0.74, "AUC score": 0.12},
        "SQUARE (5000 iterations, eps=16/255)": {"accuracy": 0.78, "attack success rate": 0.04, "AUC score": 0.86},
    },

}

plot_model_metrics_by_attack(
    data=data,
    metrics=["accuracy", "attack success rate", "AUC score"],
    output_dir="metrics_images"
)