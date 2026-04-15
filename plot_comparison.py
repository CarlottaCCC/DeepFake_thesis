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


data_white = {
    "Clean_Model": {
        "No_attack": {"accuracy": 0.902, "AUC_score": 0.964},
        "FGSM (eps=2/255)": {"accuracy": 0.068, "attack_success_rate": 0.834, "AUC_score": 0.02},
        "FGSM (eps=4/255)": {"accuracy": 0.081, "attack_success_rate": 0.821, "AUC_score": 0.02},
        "FGSM (eps=8/255)": {"accuracy": 0.151, "attack_success_rate": 0.763, "AUC_score": 0.068},
        "IFGSM": {"accuracy": 0.143, "attack_success_rate": 0.759, "AUC_score": 0.068},
        "PGD": {"accuracy": 0.013, "attack_success_rate": 0.889, "AUC_score": 0.002},
        "JSMA": {"accuracy": 0.0, "attack_success_rate": 0.902, "AUC_score": 0.0}
    },
    "FGSM-AT (eps=2/255)": {
        "No_attack": {"accuracy": 0.902, "AUC_score": 0.96},
        "FGSM (eps=2/255)": {
            "accuracy": 0.976,
            "attack_success_rate": 0.114,
            "AUC_score": 0.999
        },
       "FGSM (eps=4/255)": {
            "accuracy": 0.581,
            "attack_success_rate": 0.431,
            "AUC_score": 0.524
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.5,
            "attack_success_rate": 0.482,
            "AUC_score": 0.516
        },
        "IFGSM": {
            "accuracy": 0.087,
            "attack_success_rate": 0.815,
            "AUC_score": 0.02
        },
        "PGD": {
            "accuracy": 0.0,
            "attack_success_rate": 0.902,
            "AUC_score": 0.0
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.902, "AUC_score": 0.0}
    },
    "FGSM-AT (eps=8/255)": {
        "No_attack": {"accuracy": 0.904, "AUC_score": 0.96},
        "FGSM (eps=2/255)": {
            "accuracy": 0.886,
            "attack_success_rate": 0.164,
            "AUC_score": 0.949
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.943,
            "attack_success_rate": 0.153,
            "AUC_score": 0.989
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.979,
            "attack_success_rate": 0.113,
            "AUC_score": 0.994
        },
        "IFGSM": {
            "accuracy": 0.112,
            "attack_success_rate": 0.792,
            "AUC_score": 0.033
        },
        "PGD": {
            "accuracy": 0.0,
            "attack_success_rate": 0.904,
            "AUC_score": 0.0
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.907, "AUC_score": 0.0}
    },
    "FGSM-AT + entropy (eps=2/255)": {
        "No_attack": {"accuracy": 0.813, "AUC_score": 0.88},
        "FGSM (eps=2/255)": {
            "accuracy": 0.525,
            "attack_success_rate": 0.318,
            "AUC_score": 0.365
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.494,
            "attack_success_rate": 0.351,
            "AUC_score": 0.335
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.469,
            "attack_success_rate": 0.376,
            "AUC_score": 0.333
        },
        "IFGSM": {
            "accuracy": 0.511,
            "attack_success_rate": 0.302,
            "AUC_score": 0.311
        },
        "PGD": {
            "accuracy": 0.365,
            "attack_success_rate": 0.448,
            "AUC_score": 0.149
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.84, "AUC_score": 0.0}
    },
    "FGSM-AT + entropy (eps=8/255)": {
        "No_attack": {"accuracy": 0.823, "AUC_score": 0.87},
        "FGSM (eps=2/255)": {
            "accuracy": 0.433,
            "attack_success_rate": 0.404,
            "AUC_score": 0.283
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.414,
            "attack_success_rate": 0.433,
            "AUC_score": 0.255
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.403,
            "attack_success_rate": 0.448,
            "AUC_score": 0.255
        },
        "IFGSM": {
            "accuracy": 0.421,
            "attack_success_rate": 0.402,
            "AUC_score": 0.282
        },
        "PGD": {
            "accuracy": 0.259,
            "attack_success_rate": 0.564,
            "AUC_score": 0.091
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.841, "AUC_score": 0.0}
    },
    "FGSM-AT (cosine epsilon scheduler)": {
        "No_attack": {"accuracy": 0.827, "AUC_score": 0.89},
        "FGSM (eps=2/255)": {
            "accuracy": 0.406,
            "attack_success_rate": 0.423,
            "AUC_score": 0.356
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.368,
            "attack_success_rate": 0.515,
            "AUC_score": 0.292
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.635,
            "attack_success_rate": 0.35,
            "AUC_score": 0.61
        },
        "IFGSM": {
            "accuracy": 0.632,
            "attack_success_rate": 0.195,
            "AUC_score": 0.628
        },
        "PGD": {
            "accuracy": 0.326,
            "attack_success_rate": 0.501,
            "AUC_score": 0.24
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.892, "AUC_score": 0.0}
    },
    "FGSM-AT (linear epsilon scheduler)": {
        "No_attack": {"accuracy": 0.845, "AUC_score": 0.90},
        "FGSM (eps=2/255)": {
            "accuracy": 0.452,
            "attack_success_rate": 0.393,
            "AUC_score": 0.41
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.412,
            "attack_success_rate": 0.449,
            "AUC_score": 0.341
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.551,
            "attack_success_rate": 0.346,
            "AUC_score": 0.485
        },
        "IFGSM": {
            "accuracy": 0.693,
            "attack_success_rate": 0.152,
            "AUC_score": 0.683
        },
        "PGD": {
            "accuracy": 0.341,
            "attack_success_rate": 0.504,
            "AUC_score": 0.3
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.845, "AUC_score": 0.0}
    },
    "FGSM-AT + entropy (cosine epsilon scheduler)": {
        "No_attack": {"accuracy": 0.845, "AUC_score": 0.90},
        "FGSM (eps=2/255)": {
            "accuracy": 0.452,
            "attack_success_rate": 0.393,
            "AUC_score": 0.41
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.412,
            "attack_success_rate": 0.449,
            "AUC_score": 0.341
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.551,
            "attack_success_rate": 0.346,
            "AUC_score": 0.485
        },
        "IFGSM": {
            "accuracy": 0.693,
            "attack_success_rate": 0.152,
            "AUC_score": 0.683
        },
        "PGD": {
            "accuracy": 0.341,
            "attack_success_rate": 0.504,
            "AUC_score": 0.3
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.845, "AUC_score": 0.0}
    },
    "FGSM-AT + entropy (linear epsilon scheduler)": {
        "No_attack": {"accuracy": 0.845, "AUC_score": 0.90},
        "FGSM (eps=2/255)": {
            "accuracy": 0.452,
            "attack_success_rate": 0.393,
            "AUC_score": 0.41
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.412,
            "attack_success_rate": 0.449,
            "AUC_score": 0.341
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.551,
            "attack_success_rate": 0.346,
            "AUC_score": 0.485
        },
        "IFGSM": {
            "accuracy": 0.693,
            "attack_success_rate": 0.152,
            "AUC_score": 0.683
        },
        "PGD": {
            "accuracy": 0.341,
            "attack_success_rate": 0.504,
            "AUC_score": 0.3
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.845, "AUC_score": 0.0}
    }

}

data_black = {
    "Clean_Model": {
        "No attack": {"accuracy": 0.902, "AUC_score": 0.964},
        "SQUARE (1000 iterations, eps=16/255)": {"accuracy": 0.011, "attack_success_rate": 0.891, "AUC_score": 0.007},
        "NES": {
            "accuracy": 0.54,
            "attack_success_rate": 0.392,
            "AUC_score": 0.658
        },
        "GenAttack": {
            "accuracy":0.06,
            "attack_success_rate": 0.84,
            "AUC_score": 0.02
         }
    },
    "FGSM-AT (eps=2/255)": {
        "No attack": {"accuracy": 0.902, "AUC_score": 0.96},
        "SQUARE": {
            "accuracy": 0.001,
            "attack_success_rate": 0.901,
            "AUC_score": 0.001
        },
        "NES": {
            "accuracy": 0.541,
            "attack_success_rate": 0.445,
            "AUC_score": 0.573
        },
        "GenAttack": {
            "accuracy": 0.0,
            "attack_success_rate": 0.902,
            "AUC_score": 0.0
        }
    },
    "FGSM-AT (eps=8/255)": {
        "No attack": {"accuracy": 0.904, "AUC_score": 0.96},
        "SQUARE": {
            "accuracy": 0.004,
            "attack_success_rate": 0.9,
            "AUC_score": 0.003
        },
        "NES": {
            "accuracy": 0.611,
            "attack_success_rate": 0.347,
            "AUC_score": 0.668
        },
        "GenAttack": {
            "accuracy": 0.0,
            "attack_success_rate": 0.904,
            "AUC_score": 0.0
        }
    },
    "FGSM-AT + entropy (eps=2/255)": {
        "No attack": {"accuracy": 0.813, "AUC_score": 0.88},
        "SQUARE": {
            "accuracy": 0.072,
            "attack_success_rate": 0.741,
            "AUC_score": 0.076
        },
        "NES": {
            "accuracy": 0.808,
            "attack_success_rate": 0.209,
            "AUC_score": 0.848
        },
        "GenAttack": {
            "accuracy": 0.431,
            "attack_success_rate": 0.382,
            "AUC_score": 0.232
        }
    },
    "FGSM-AT + entropy (eps=8/255)": {
        "No attack": {"accuracy": 0.823, "AUC_score": 0.87},
        "SQUARE": {
            "accuracy": 0.056,
            "attack_success_rate": 0.767,
            "AUC_score": 0.071
        },
        "NES": {
            "accuracy": 0.569,
            "attack_success_rate": 0.38,
            "AUC_score": 0.61
        },
        "GenAttack": {
            "accuracy": 0.082,
            "attack_success_rate": 0.759,
            "AUC_score": 0.048
        }
    },
    "FGSM-AT (cosine epsilon scheduler)": {
        "No_attack": {"accuracy": 0.827, "AUC_score": 0.89},
        "SQUARE": {
            "accuracy": 0.023,
            "attack_success_rate": 0.804,
            "AUC_score": 0.018
        },
        "GenAttack": {
            "accuracy": 0.446,
            "attack_success_rate": 0.381,
            "AUC_score": 0.577
        },
        "NES": {
            "accuracy": 0.588,
            "attack_success_rate": 0.369,
            "AUC_score": 0.655
        }
    },
    "FGSM-AT (linear epsilon scheduler)": {
        "No_attack": {"accuracy": 0.845, "AUC_score": 0.90},
        "SQUARE": {
            "accuracy": 0.027,
            "attack_success_rate": 0.818,
            "AUC_score": 0.027
        },
        "GenAttack": {
            "accuracy": 0.468,
            "attack_success_rate": 0.377,
            "AUC_score": 0.568
        },
        "NES": {
            "accuracy": 0.614,
            "attack_success_rate": 0.339,
            "AUC_score": 0.682
        }
    },
    "FGSM-AT + entropy (cosine epsilon scheduler)": {
        "No_attack": {"accuracy": 0.845, "AUC_score": 0.90},
        "FGSM (eps=2/255)": {
            "accuracy": 0.452,
            "attack_success_rate": 0.393,
            "AUC_score": 0.41
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.412,
            "attack_success_rate": 0.449,
            "AUC_score": 0.341
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.551,
            "attack_success_rate": 0.346,
            "AUC_score": 0.485
        },
        "IFGSM": {
            "accuracy": 0.693,
            "attack_success_rate": 0.152,
            "AUC_score": 0.683
        },
        "PGD": {
            "accuracy": 0.341,
            "attack_success_rate": 0.504,
            "AUC_score": 0.3
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.845, "AUC_score": 0.0}
    },
    "FGSM-AT + entropy (linear epsilon scheduler)": {
        "No_attack": {"accuracy": 0.845, "AUC_score": 0.90},
        "FGSM (eps=2/255)": {
            "accuracy": 0.452,
            "attack_success_rate": 0.393,
            "AUC_score": 0.41
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.412,
            "attack_success_rate": 0.449,
            "AUC_score": 0.341
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.551,
            "attack_success_rate": 0.346,
            "AUC_score": 0.485
        },
        "IFGSM": {
            "accuracy": 0.693,
            "attack_success_rate": 0.152,
            "AUC_score": 0.683
        },
        "PGD": {
            "accuracy": 0.341,
            "attack_success_rate": 0.504,
            "AUC_score": 0.3
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.845, "AUC_score": 0.0}
    }
}
 
data_old = {
    "Clean_Model": {
        "No_attack": {"accuracy": 0.902, "AUC_score": 0.964},
        "FGSM (eps=2/255)": {"accuracy": 0.068, "attack_success_rate": 0.834, "AUC_score": 0.02},
        "FGSM (eps=4/255)": {"accuracy": 0.081, "attack_success_rate": 0.821, "AUC_score": 0.02},
        "FGSM (eps=8/255)": {"accuracy": 0.151, "attack_success_rate": 0.763, "AUC_score": 0.068},
        "SQUARE (1000 iterations, eps=16/255)": {"accuracy": 0.011, "attack_success_rate": 0.891, "AUC_score": 0.007},
        
    },
    "FGSM-AT (eps=2/255)": {
        "No_attack": {"accuracy": 0.902, "AUC_score": 0.96},
        "FGSM (eps=2/255)": {
            "accuracy": 0.976,
            "attack_success_rate": 0.114,
            "AUC_score": 0.999
        },
       "FGSM (eps=4/255)": {
            "accuracy": 0.581,
            "attack_success_rate": 0.431,
            "AUC_score": 0.524
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.5,
            "attack_success_rate": 0.482,
            "AUC_score": 0.516
        },
        "SQUARE": {
            "accuracy": 0.001,
            "attack_success_rate": 0.901,
            "AUC_score": 0.001
        }
        
    },
    "FGSM-AT + entropy (eps=2/255)": {
        "No_attack": {"accuracy": 0.813, "AUC_score": 0.88},
        "FGSM (eps=2/255)": {
            "accuracy": 0.525,
            "attack_success_rate": 0.318,
            "AUC_score": 0.365
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.494,
            "attack_success_rate": 0.351,
            "AUC_score": 0.335
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.469,
            "attack_success_rate": 0.376,
            "AUC_score": 0.333
        },
        "SQUARE": {
            "accuracy": 0.072,
            "attack_success_rate": 0.741,
            "AUC_score": 0.076
        }
    }
}


data_l = {
        "FGSM (eps=2/255)": {"avg_l2": 3.004, "avg_linf": 0.007},
        "FGSM (eps=4/255)": {"avg_l2": 5.98, "avg_linf": 0.01},
        "FGSM (eps=8/255)": {"avg_l2": 11.91, "avg_linf": 0.03},
        "IFGSM (eps=8/255)": {"avg_l2": 0.65, "avg_linf": 0.002},
        "PGD (eps=8/255)": {"avg_l2": 6.91, "avg_linf": 0.03},
        "JSMA": {"avg_l2": 0.74, "avg_linf": 0.29},
        "SQUARE": {"avg_l2": 20.89, "avg_linf": 0.05},
        "GenAttack": {"avg_l2": 7.52, "avg_linf": 0.03},
        "NES": {"avg_l2": 85.04, "avg_linf": 0.49},
        #"ZOO": {"avg_l2": 0.068, "avg_linf": 0.834},
        #"AutoZOOM": {"avg_l2": 0.068, "avg_linf": 0.834}
}

df_l = pd.DataFrame(data_l).T.reset_index()
df_l.rename(columns={"index": "attack"}, inplace=True)
#barplot_perturbation(df_l)

# Plot ASR
plot_metric_bar(data_old, metric="attack_success_rate", log_scale=False, save_path="comparison_images/ASR_comparison_by_attack_b.png")
## Plot accuracy
plot_metric_bar(data_old, metric="accuracy", log_scale=False, save_path="comparison_images/accuracy_comparison_by_attack_b.png")
## AUC
plot_metric_bar(data_old, metric="AUC_score", log_scale=False, save_path="comparison_images/AUC_by_attack_b.png")

#plot_model_metrics_heatmap(
#    data_white,
#    metrics=["accuracy", "attack_success_rate", "AUC_score"],
#    output_dir="comparison_images",
#    attack_type="white"
#)

#df_l = pd.DataFrame(data_l).T
#df_l.rename(columns={"index": "attack"}, inplace=True)
#
## Log scale per L2, Linf lasciamo lineare (puoi anche fare log se vuoi)
#df_plot = df_l.copy()
#df_plot['avg_l2'] = np.log10(df_plot['avg_l2'])  # log10 scale
#
## Heatmap
#plt.figure(figsize=(10,6))
#sns.heatmap(df_plot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Log10(L2) / Linf'})
#plt.title("Average perturbations per attack (L2 in log scale)")
#plt.ylabel("Attack")
#plt.xlabel("Metric")
#plt.tight_layout()
#
## Salvataggio file
#plt.savefig("heatmap_perturbations_logL2.png", dpi=300, bbox_inches="tight", facecolor='white')
#plt.show()
