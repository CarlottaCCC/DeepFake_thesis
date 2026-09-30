from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import pandas as pd
import seaborn as sns

def barplot_perturbation(df_l):
    df_l = pd.DataFrame(data_l).T.reset_index()
    df_l.rename(columns={"index": "attack"}, inplace=True)
    
    # Ordina per avg_l2 decrescente
    df_l = df_l.sort_values(by="avg_l2", ascending=False)
    
    # Melt per barplot affiancato
    df_melted = df_l.melt(id_vars="attack", value_vars=["avg_l2", "avg_linf"], 
                          var_name="metric", value_name="value")
    
    # Plot
    plt.figure(figsize=(14,6))
    ax = sns.barplot(data=df_melted, x="attack", y="value", hue="metric", edgecolor="black")
    
    # Imposta scala logaritmica
    ax.set_yscale('log')
    
    # Aggiungi valori sopra le barre (anche se su scala log possono essere piccoli)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # evita log(0)
            ax.annotate(f'{height:.3f}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average perturbation (log scale)")
    plt.title("Average L2 and Linf comparison per attack (log scale)")
    plt.tight_layout()
    
    # Salvataggio
    plt.savefig("comparison_images/avg_perturbation_barplot_log_2.png", dpi=300, bbox_inches="tight", facecolor='white')


fgsm_models_names = [{"model_label": "FGSM-AT + entropy eps 2", "model_name":"no_eps_scheduler/resnet50_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42.pt"},
                             {"model_label": "FGSM-AT eps 2", "model_name":"no_eps_scheduler/resnet50_fgsm_epoch_24_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42.pt"},
                             {"model_label": "FGSM-AT + entropy eps 8", "model_name":"no_eps_scheduler/resnet50_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42.pt"},
                             {"model_label": "FGSM-AT eps 8", "model_name":"no_eps_scheduler/resnet50_fgsm_epoch_24_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42.pt"},
                             {"model_label":"FGSM-AT cosine sched", "model_name":"with_eps_scheduler_fgsm/resnet50_fgsm_epoch_24_LR_0.0001_batchsize_32_WD_0.01_cosine_eps_scheduler_seed_42.pt"},
                             {"model_label":"FGSM-AT linear sched", "model_name":"with_eps_scheduler_fgsm/resnet50_fgsm_epoch_24_LR_0.0001_batchsize_32_WD_0.01_linear_eps_scheduler_seed_42.pt"},
                             {"model_label":"FGSM-AT + entropy cosine sched", "model_name":"with_eps_scheduler_fgsm/resnet50_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_cosine_eps_scheduler_seed_42.pt"},
                             {"model_label":"FGSM-AT + entropy linear sched", "model_name":"with_eps_scheduler_fgsm/resnet50_square_epoch_24_LR_0.0001_batchsize_32_WD_0.01_linear_eps_scheduler_seed_42.pt"},
                             {"model_label":"PGD-AT cosine sched", "model_name":"pgdat_ades_difat/resnet50_pgdat_baseline__cosine_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25.pt"}]

for model_item in fgsm_models_names:

    model_label = model_item["model_label"]
    model_name = model_item["model_name"]
    results_path = f"ades_difat/test_black_box_results/{model_label}"