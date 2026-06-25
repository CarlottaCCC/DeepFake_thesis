import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from test_generic import test_attack
from train_clean import train_clean_f
from train_robust_FGSM import train_robust
from train_robust_SQUARE import train_robust_with_entropy

#if __name__ == "__main__":

