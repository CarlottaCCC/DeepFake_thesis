import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
val_metrics_adv = Metrics()

#models_no_eps_sched= {
#    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_2": "history_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_2.pt",
#    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_2": "history_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_2.pt",
#    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_2": "history_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_2.pt",
#    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_2": "history_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_2.pt"
#}
#
#models_eps_sched = {
#    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_2": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_2.pt",
#    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_2": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_2.pt",
#    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_2": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_2.pt",
#    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_2": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_2.pt",
#    #"FGSM-AT + entropy (eps=2/255)": "resnet50_square_epoch_18_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196.pt",
#    #"FGSM-AT + entropy (eps=8/255)": "resnet50_square_epoch_13_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784.pt",
#}

fgsm_history = {
    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init": "with_eps_scheduler/history_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init.json",
    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init": "with_eps_scheduler/history_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init.json",
    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_4_init": "no_eps_scheduler/history_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_4_init.json",
    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_4_init": "no_eps_scheduler/history_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_4_init.json"
}

square_history = {
    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_4_init": "no_eps_scheduler/history_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_4_init.json",
    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_4_init": "no_eps_scheduler/history_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_4_init.json",
    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init": "with_eps_scheduler/history_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init.json",
    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init": "with_eps_scheduler/history_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init.json"

}

fgsm_models = {
    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init": "with_eps_scheduler/resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init.pt",
    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init": "with_eps_scheduler/resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init.pt",
    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_4_init": "no_eps_scheduler/resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_4_init.pt",
    "fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_4_init": "no_eps_scheduler/resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_4_init.pt"
}

square_models = {
    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_4_init": "no_eps_scheduler/resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_4_init.pt",
    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_4_init": "no_eps_scheduler/resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_4_init.pt",
    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init": "with_eps_scheduler/resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init.pt",
    "square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init": "with_eps_scheduler/resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init.pt"

}
device = "cuda"
#for file_name, model_name in fgsm_models.items():
#    checkpoint_path = f"{MODELS_DIR}/{model_name}"
#    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#    val_auc         = checkpoint['val_auc_adv']
#    fpr = checkpoint["val_fpr_adv"]
#    tpr = checkpoint["val_tpr_adv"]
#    plot_roc(fpr, tpr, val_auc[11], 12, 
#                     f"plots/fgsm/ROC_plot_{file_name}.png")

for file_name, model_name in square_models.items():
    device = "cuda"
    checkpoint_path = f"{MODELS_DIR}/{model_name}"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    val_auc         = checkpoint['val_auc_adv']
    fpr = checkpoint["val_fpr_adv"]
    tpr = checkpoint["val_tpr_adv"]
    plot_roc(fpr, tpr, val_auc[11], 12, 
                     f"plots/square/ROC_plot_{file_name}.png")

#for file_name, history_name in fgsm_history.items():
#    with open(f"history/history_fgsm/{history_name}", "r") as f:
#        history = json.load(f)
#
#        folder_name = "fgsm"
#
#        train_metrics_clean.f1_list = history["train_f1_clean"]
#        train_metrics_adv.f1_list = history["train_f1_adv"]
#        val_metrics_adv.f1_list = history["val_f1_adv"]
#        
#        train_metrics_clean.precision_list = history["train_precision_clean"]
#        train_metrics_adv.precision_list = history["train_precision_adv"]
#        val_metrics_adv.precision_list = history["val_precision_adv"]
#        
#        train_metrics_clean.recall_list = history["train_recall_clean"]
#        train_metrics_adv.recall_list = history["train_recall_adv"]
#        val_metrics_adv.recall_list = history["val_recall_adv"]
#        
#        train_metrics_clean.accuracy_list = history["train_accuracy_clean"]
#        train_metrics_adv.accuracy_list = history["train_accuracy_adv"]
#        val_metrics_adv.accuracy_list = history["val_accuracy_adv"]
#        
#        train_metrics_clean.train_losses = history["train_losses"]
#
#        plot_metric(train_metrics_adv.accuracy_list, val_metrics_adv.accuracy_list, 12, "Accuracy", 
#           f"plots/{folder_name}/Accuracy_plot_{file_name}.png")
#        #plot f1 score
#        #plot_metric(train_metrics_adv.f1_list,  val_metrics_adv.f1_list, 12, "F1_score", 
#        #            f"plots/{folder_name}/F1_plot_{file_name}.png")
#        ##plot precision
#        #plot_metric(train_metrics_adv.precision_list,  val_metrics_adv.precision_list, 12, "Precision",
#        #            f"plots/{folder_name}/Precision_plot_{file_name}.png")
#        ##plot recall
#        #plot_metric(train_metrics_adv.recall_list,  val_metrics_adv.recall_list, 12, "Recall", 
#        #            f"plots/{folder_name}/Recall_plot_{file_name}.png")
#        
#        plot_loss(train_metrics_clean.train_losses, f"plots/{folder_name}/Loss_plot_{file_name}.png")
##
#for file_name, history_name in square_history.items():
#    with open(f"history/history_square/{history_name}", "r") as f:
#        history = json.load(f)
#
#        folder_name = "square"
#
#        train_metrics_clean.f1_list = history["train_f1_clean"]
#        train_metrics_adv.f1_list = history["train_f1_adv"]
#        val_metrics_adv.f1_list = history["val_f1_adv"]
#        
#        train_metrics_clean.precision_list = history["train_precision_clean"]
#        train_metrics_adv.precision_list = history["train_precision_adv"]
#        val_metrics_adv.precision_list = history["val_precision_adv"]
#        
#        train_metrics_clean.recall_list = history["train_recall_clean"]
#        train_metrics_adv.recall_list = history["train_recall_adv"]
#        val_metrics_adv.recall_list = history["val_recall_adv"]
#        
#        train_metrics_clean.accuracy_list = history["train_accuracy_clean"]
#        train_metrics_adv.accuracy_list = history["train_accuracy_adv"]
#        val_metrics_adv.accuracy_list = history["val_accuracy_adv"]
#        
#        train_metrics_clean.train_losses = history["train_losses"]
#
#        plot_metric(train_metrics_adv.accuracy_list, val_metrics_adv.accuracy_list, 12, "Accuracy", 
#           f"plots/{folder_name}/Accuracy_plot_{file_name}.png")
#        
#        #plot f1 score
#        #plot_metric(train_metrics_adv.f1_list,  val_metrics_adv.f1_list, 12, "F1_score", 
#        #            f"plots/{folder_name}/F1_plot_{file_name}.png")
#        ##plot precision
#        #plot_metric(train_metrics_adv.precision_list,  val_metrics_adv.precision_list, 12, "Precision",
#        #            f"plots/{folder_name}/Precision_plot_{file_name}.png")
#        ##plot recall
#        #plot_metric(train_metrics_adv.recall_list,  val_metrics_adv.recall_list, 12, "Recall", 
#        #            f"plots/{folder_name}/Recall_plot_{file_name}.png")
#        
#        plot_loss(train_metrics_clean.train_losses, f"plots/{folder_name}/Loss_plot_{file_name}.png")
##
###train_metrics_clean.f1_list = history["train_f1"]
###val_metrics.f1_list = history["val_f1"]
###
##train_metrics_clean.precision_list = history["train_precision"]
##val_metrics.precision_list = history["val_precision"]
#
#train_metrics_clean.recall_list = history["train_recall"]
#val_metrics.recall_list = history["val_recall"]
#
#train_metrics_clean.accuracy_list = history["train_accuracy"]
#val_metrics.accuracy_list = history["val_accuracy"]
#
#train_metrics_clean.train_losses = history["train_losses"]

#plot_metric(train_metrics_clean.accuracy_list, val_metrics.accuracy_list, 12, "Accuracy", 
#            f"plots/fgsm/Accuracy_plot_fgsm_epoch_14_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_fine_tuned.png")
##plot f1 score
#plot_metric(train_metrics_clean.f1_list,  val_metrics.f1_list, 12, "F1_score", 
#            f"plots/fgsm/F1_plot_fgsm_epoch_14_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_fine_tuned.png")
##plot precision
#plot_metric(train_metrics_clean.precision_list,  val_metrics.precision_list, 12, "Precision",
#            f"plots/fgsm/Precision_plot_fgsm_epoch_14_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_fine_tuned.png")
##plot recall
#plot_metric(train_metrics_clean.recall_list,  val_metrics.recall_list, 12, "Recall", 
#            f"plots/fgsm/Recall_plot_fgsm_epoch_14_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_fine_tuned.png")
#
#plot_loss(train_metrics_clean.train_losses, f"plots/fgsm/Loss_plot_fgsm_epoch_14_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_fine_tuned.png")






#checkpoint_path = "models/clean_resnet50/resnet50_clean_epoch_20_LR_0.0003_batchsize_32_WD_1e-05.pt"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#val_metrics.auc_list = checkpoint["val_auc"]
#val_metrics.tpr = checkpoint["val tpr"]
#val_metrics.fpr = checkpoint["val fpr"]
##plot AUC
#plot_roc(val_metrics.fpr, val_metrics.tpr, val_metrics.auc_list[NUM_EPOCHS-1], NUM_EPOCHS, 
#         f"{root}/Train_ROC_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")
