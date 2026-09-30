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

import matplotlib.pyplot as plt


def plot_model_metrics(metrics, model_name="Model", save_path=None):

    names = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        names,
        values,
        width=0.6
    )

    # Add values above bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_title(
        f"{model_name} - Performance Metrics",
        fontsize=14
    )

    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    if save_path is not None:
        directory = os.path.dirname(save_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

import os
import matplotlib.pyplot as plt


def plot_training_validation_loss(
    train_loss,
    val_loss,
    model_name="Model",
    save_path=None
):

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        train_loss,
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training and Validation Loss")

    plt.legend()
    plt.grid(
        axis="both",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    if save_path is not None:
        directory = os.path.dirname(save_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

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

fgsm_models_names = [{"model_label": "FGSM-AT + entropy eps 2", "model_name":"history_square/no_eps_scheduler/history_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42.json"},
                             {"model_label": "FGSM-AT eps 2", "model_name":"history_fgsm/no_eps_scheduler/history_fgsm_epoch_24_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42.json"},
                             {"model_label": "FGSM-AT + entropy eps 8", "model_name":"history_square/no_eps_scheduler/history_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42.json"},
                             {"model_label": "FGSM-AT eps 8", "model_name":"history_fgsm/no_eps_scheduler/history_fgsm_epoch_24_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42.json"},
                             {"model_label":"FGSM-AT cosine sched", "model_name":"history_fgsm/with_eps_scheduler/history_fgsm_epoch_24_LR_0.0001_batchsize_32_WD_0.01_cosine_eps_scheduler_seed_42.json"},
                             {"model_label":"FGSM-AT linear sched", "model_name":"history_fgsm/with_eps_scheduler/history_fgsm_epoch_24_LR_0.0001_batchsize_32_WD_0.01_linear_eps_scheduler_seed_42.json"},
                             {"model_label":"FGSM-AT + entropy cosine sched", "model_name":"history_square/with_eps_scheduler/history_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_cosine_eps_scheduler_seed_42.json"},
                             {"model_label":"FGSM-AT + entropy linear sched", "model_name":"history_square/with_eps_scheduler/history_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_linear_eps_scheduler_seed_42.json"}]

square_models_names = [{"model_label": "FGSM-AT + entropy eps 2", "model_name":"history_square/no_eps_scheduler/history_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_with_val.json"},
                             {"model_label": "FGSM-AT + entropy eps 8", "model_name":"history_square/no_eps_scheduler/history_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_with_val.json"},
                             {"model_label":"FGSM-AT + entropy cosine sched", "model_name":"history_square/with_eps_scheduler/history_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_cosine_eps_scheduler_seed_42_with_val.json"},
                             {"model_label":"FGSM-AT + entropy linear sched", "model_name":"history_square/with_eps_scheduler/history_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_linear_eps_scheduler_seed_42_with_val.json"}]

pgdat_models_names = [{"model_label":"PGD-AT linear eps sched", "model_name":"pgdat_ades_difat/resnet50_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25_freeze.pt"},
                            {"model_label":"ADES", "model_name":"pgdat_ades_difat/resnet50_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_50__lr_0.001_seed_42_epochs_25_freeze_2.pt"},
                            {"model_label":"ADES + 11 epochs PGD-AT", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_ades_baseline_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"PGD-AT + DiFat conf 2", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_7__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"PGD-AT + DiFat conf 4", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_[3, 6, 13]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"ADES + PGD-AT + DiFat", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_ades_baseline_difat_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23]_epochsdifat_[24]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"PGD-AT cosine sched", "model_name":"pgdat_ades_difat/resnet50_pgdat_baseline__cosine_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25.pt"}]

pgdat_models_history = [{"model_label":"PGD-AT linear eps sched", "model_name":"history_pgdat_ades_difat/history_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25_freeze.json"},
                            {"model_label":"ADES", "model_name":"history_pgdat_ades_difat/history_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_50__lr_0.001_seed_42_epochs_25_freeze__norampup.json"},
                            {"model_label":"ADES + 11 epochs PGD-AT", "model_name":"history_pgdat_mixed_trainings/history_pgdat_ades_baseline_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]__lr_0.001_seed_42_epochs_25_numeprampup_12.json"},
                            {"model_label":"PGD-AT + DiFat conf 2", "model_name":"history_pgdat_mixed_trainings/history_pgdat_baseline_difat_epochsdifat_7__lr_0.001_seed_42_epochs_25_numeprampup_12.json"},
                            {"model_label":"PGD-AT + DiFat conf 4", "model_name":"history_pgdat_mixed_trainings/history_pgdat_baseline_difat_epochsdifat_[3, 6, 13]__lr_0.001_seed_42_epochs_25_numeprampup_12.json"},
                            {"model_label":"ADES + PGD-AT + DiFat", "model_name":"history_pgdat_mixed_trainings/history_pgdat_ades_baseline_difat_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23]_epochsdifat_[24]__lr_0.001_seed_42_epochs_25_numeprampup_12.json"},
                            {"model_label":"PGD-AT cosine sched", "model_name":"history_pgdat_ades_difat/history_resnet50_pgdat_baseline__cosine_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25.json"}]

clean_model_history = [{"model_label":"baseline model", "model_name":"history_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_DROPOUT_0.0_aug.json"}]
clean_model_name = [{"model_label":"baseline model", "model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"}]
device = "cuda"

for model_values in clean_model_history:

    model_label = model_values["model_label"]
    model_name = f"history/history_clean/{model_values['model_name']}"

    with open(model_name, "r") as f:
        data = json.load(f)

    #plot_training_validation_loss(
    #    data["train_losses"],
    #    data["val_losses"],
    #    model_name=model_label,
    #    save_path=f"plots_for_thesis/{model_label}/{model_label}_loss.png"
    #)


    #metrics = {
    #            "Clean Acc": data["train_accuracy_clean"][24],
    #            "Adv Acc": data["train_accuracy_adv"][24],
    #            "Clean Prec": data["train_precision_clean"][24],
    #            "Adv Prec": data["train_precision_adv"][24],
    #            "Clean Recall": data["train_recall_clean"][24],
    #            "Adv Recall": data["train_recall_adv"][24],
    #            "Clean F1": data["train_f1_clean"][24],
    #            "Adv F1": data["train_f1_adv"][24]
    #        }

    #metrics = {
    #                "Clean Acc": data["train_accuracy"][11],
    #                "Clean Prec": data["train_precision"][11],
    #                "Clean Recall": data["train_recall"][11],
    #                "Clean F1": data["train_f1"][11],
    #                
    #            }
#
    #plot_model_metrics(
    #    metrics,
    #    model_name=model_label,
    #    save_path=f"plots_for_thesis/{model_label}/{model_label}_metrics.png"
    #)

for model_values in pgdat_models_names:

    # ROC CURVE
    model_label = model_values["model_label"]
    model_name = f"{MODELS_DIR}/{model_values['model_name']}"
    checkpoint_path = model_name
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(model_label)
    val_auc = checkpoint['val_auc_adv']
    fpr = checkpoint["val_fpr_adv"]
    tpr = checkpoint["val_tpr_adv"]
    plot_roc(fpr, tpr, val_auc[24], 25, 
                     f"plots_for_thesis/{model_label}/{model_label}_ROC_plot_val_adv.png")

