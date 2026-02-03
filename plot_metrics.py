from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

train_metrics_clean = Metrics()
train_metrics_adv = Metrics()
val_metrics = Metrics()

root = "metrics_images_clean"

#with open("history_clean/history_clean_epoch_20_LR_0.0003_batchsize_32_WD_1e-05.json", "r") as f:
#    history = json.load(f)
#
#
#
#train_metrics_clean.f1_list = history["train_f1_clean"]
#train_metrics_adv.f1_list = history["train_f1_adv"]
#val_metrics.f1_list = history["val_f1"]
#
#train_metrics_clean.precision_list = history["train_precision_clean"]
#train_metrics_adv.precision_list = history["train_precision_adv"]
#val_metrics.precision_list = history["val_precision"]
#
#train_metrics_clean.recall_list = history["train_recall_clean"]
#train_metrics_adv.recall_list = history["train_recall_adv"]
#val_metrics.recall_list = history["val_recall"]
#
#train_metrics_clean.accuracy_list = history["train_accuracy_clean"]
#train_metrics_adv.accuracy_list = history["train_accuracy_adv"]
#val_metrics.accuracy_list = history["val_accuracy"]
#
#
##plot accuracy
#plot_metric(train_metrics_clean.accuracy_list, val_metrics.accuracy_list, NUM_EPOCHS, "Accuracy", 
#            f"{root}/Train_accuracy_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")
##plot f1 score
#plot_metric(train_metrics_clean.f1_list,  val_metrics.f1_list, NUM_EPOCHS, "F1_score", 
#            f"{root}/Train_F1_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")
##plot precision
#plot_metric(train_metrics_clean.precision_list,  val_metrics.precision_list, NUM_EPOCHS, "Precision",
#            f"{root}/Train_precision_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")
##plot recall
#plot_metric(train_metrics_clean.recall_list,  val_metrics.recall_list, NUM_EPOCHS, "Recall", 
#            f"{root}/Train_recall_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")

checkpoint_path = "models/clean_resnet50/resnet50_clean_epoch_20_LR_0.0003_batchsize_32_WD_1e-05.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
val_metrics.auc_list = checkpoint["val_auc"]
val_metrics.tpr = checkpoint["val tpr"]
val_metrics.fpr = checkpoint["val fpr"]
#plot AUC
plot_roc(val_metrics.fpr, val_metrics.tpr, val_metrics.auc_list[NUM_EPOCHS-1], NUM_EPOCHS, 
         f"{root}/Train_ROC_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")