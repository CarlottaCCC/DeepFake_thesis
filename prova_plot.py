from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


with open("history_clean/history_clean_training_batchsize_32_LR_0.0001_WD_0.0001.json", "r") as f:
    history = json.load(f)

#plot loss
plot_loss(history["train_losses"])
#plot accuracy
plot_metric(history["train_accuracy"], history["val_accuracy"], "Accuracy")
#plot f1 score
plot_metric(history["train_f1"], history["val_f1"], "F1_score")
#plot precision
plot_metric(history["train_precision"], history["val_precision"], "Precision")
#plot recall
plot_metric(history["train_recall"], history["val_recall"], "Recall")
#plot AUC
########
