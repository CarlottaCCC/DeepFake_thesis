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
        "No_attack": {"accuracy": 0.906, "AUC_score": 0.97},
        "FGSM (eps=2/255)": {
            "accuracy": 0.992,
            "attack_success_rate": 0.098,
            "AUC_score": 1.0
        },
       "FGSM (eps=4/255)": {
            "accuracy": 0.466,
            "attack_success_rate": 0.488,
            "AUC_score": 0.387
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.5,
            "attack_success_rate": 0.456,
            "AUC_score": 0.383
        },
        "IFGSM": {
            "accuracy": 0.091,
            "attack_success_rate": 0.815,
            "AUC_score": 0.031
        },
        "PGD": {
            "accuracy": 0.0,
            "attack_success_rate": 0.906,
            "AUC_score": 0.0
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.906, "AUC_score": 0.0}
    },
    "FGSM-AT (eps=8/255)": {
        "No_attack": {"accuracy": 0.907, "AUC_score": 0.96},
         "FGSM (eps=2/255)": {
            "accuracy": 0.073,
            "attack_success_rate": 0.834,
            "AUC_score": 0.027
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.516,
            "attack_success_rate": 0.499,
            "AUC_score": 0.709
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.99,
            "attack_success_rate": 0.103,
            "AUC_score": 1.0
        },
        "IFGSM": {
            "accuracy": 0.21,
            "attack_success_rate": 0.697,
            "AUC_score": 0.114
        },
        "PGD": {
            "accuracy": 0.006,
            "attack_success_rate": 0.901,
            "AUC_score": 0.0
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.907, "AUC_score": 0.0}
    },
    "FGSM-AT + entropy (eps=2/255)": {
        "No_attack": {"accuracy": 0.911, "AUC_score": 0.97},
        "FGSM (eps=2/255)": {
            "accuracy": 0.997,
            "attack_success_rate": 0.092,
            "AUC_score": 1.0
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.508,
            "attack_success_rate": 0.469,
            "AUC_score": 0.465
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.419,
            "attack_success_rate": 0.562,
            "AUC_score": 0.333
        },
        "IFGSM": {
            "accuracy": 0.123,
            "attack_success_rate": 0.788,
            "AUC_score": 0.035
        },
        "PGD": {
            "accuracy": 0.0,
            "attack_success_rate": 0.911,
            "AUC_score": 0.0
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.911, "AUC_score": 0.0}
    },
    "FGSM-AT + entropy (eps=8/255)": {
        "No_attack": {"accuracy": 0.913, "AUC_score": 0.968},
        "FGSM (eps=2/255)": {
            "accuracy": 0.087,
            "attack_success_rate": 0.826,
            "AUC_score": 0.019
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.161,
            "attack_success_rate": 0.762,
            "AUC_score": 0.022
        },
        "FGSM (eps=8/255)": {
            "accuracy": 1.0,
            "attack_success_rate": 0.087,
            "AUC_score": 1.0
        },
        "IFGSM": {
            "accuracy": 0.219,
            "attack_success_rate": 0.694,
            "AUC_score": 0.112
        },
        "PGD": {
            "accuracy": 0.036,
            "attack_success_rate": 0.877,
            "AUC_score": 0.0
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.913, "AUC_score": 0.0}
    },
    "FGSM-AT (cosine epsilon scheduler)": {
        "No_attack": {"accuracy": 0.86, "AUC_score": 0.93},
        "FGSM (eps=2/255)": {
            "accuracy": 0.257,
            "attack_success_rate": 0.608,
            "AUC_score": 0.196
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.21,
            "attack_success_rate": 0.661,
            "AUC_score": 0.116
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.978,
            "attack_success_rate": 0.131,
            "AUC_score": 0.999
        },
        "IFGSM": {
            "accuracy": 0.611,
            "attack_success_rate": 0.254,
            "AUC_score": 0.629
        },
        "PGD": {
            "accuracy": 0.244,
            "attack_success_rate": 0.621,
            "AUC_score": 0.112
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.86, "AUC_score": 0.0}
    },
    "FGSM-AT (linear epsilon scheduler)": {
        "No_attack": {"accuracy": 0.782, "AUC_score": 0.87},
        "FGSM (eps=2/255)": {
            "accuracy": 0.44,
            "attack_success_rate": 0.342,
            "AUC_score": 0.452
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.342,
            "attack_success_rate": 0.444,
            "AUC_score": 0.289
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.287,
            "attack_success_rate": 0.519,
            "AUC_score": 0.191
        },
        "IFGSM": {
            "accuracy": 0.638,
            "attack_success_rate": 0.144,
            "AUC_score": 0.706
        },
        "PGD": {
            "accuracy": 0.473,
            "attack_success_rate": 0.309,
            "AUC_score": 0.496
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.78, "AUC_score": 0.0}
    },
    "FGSM-AT + entropy (cosine epsilon scheduler)": {
        "No_attack": {"accuracy": 0.82, "AUC_score": 0.89},
        "FGSM (eps=2/255)": {
            "accuracy": 0.325,
            "attack_success_rate": 0.503,
            "AUC_score": 0.335
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.347,
            "attack_success_rate": 0.491,
            "AUC_score": 0.282
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.456,
            "attack_success_rate": 0.42,
            "AUC_score": 0.411
        },
        "IFGSM": {
            "accuracy": 0.6,
            "attack_success_rate": 0.228,
            "AUC_score": 0.611
        },
        "PGD": {
            "accuracy": 0.243,
            "attack_success_rate": 0.585,
            "AUC_score": 0.193
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.82, "AUC_score": 0.0}
    },
    "FGSM-AT + entropy (linear epsilon scheduler)": {
        "No_attack": {"accuracy": 0.81, "AUC_score": 0.88},
        "FGSM (eps=2/255)": {
            "accuracy": 0.479,
            "attack_success_rate": 0.333,
            "AUC_score": 0.466
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.388,
            "attack_success_rate": 0.426,
            "AUC_score": 0.314
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.377,
            "attack_success_rate": 0.449,
            "AUC_score": 0.246
        },
        "IFGSM": {
            "accuracy": 0.665,
            "attack_success_rate": 0.147,
            "AUC_score": 0.703
        },
        "PGD": {
            "accuracy": 0.487,
            "attack_success_rate": 0.325,
            "AUC_score": 0.486
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.84, "AUC_score": 0.0}
    }

}

data_black = {
    "Clean_Model": {
        "No_attack": {"accuracy": 0.902, "AUC_score": 0.964},
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
        "No_attack": {"accuracy": 0.906, "AUC_score": 0.97},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.00,
            "attack_success_rate": 0.906,
            "AUC_score": 0.00
        },
        "NES": {
            "accuracy": 0.504,
            "attack_success_rate": 0.448,
            "AUC_score": 0.502
        },
        "GenAttack": {
            "accuracy": 0.0,
            "attack_success_rate": 0.906,
            "AUC_score": 0.0
        }
    },
    "FGSM-AT (eps=8/255)": {
        "No_attack": {"accuracy": 0.907, "AUC_score": 0.96},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.011,
            "attack_success_rate": 0.89,
            "AUC_score": 0.006
        },
        "NES": {
            "accuracy": 0.656,
            "attack_success_rate": 0.329,
            "AUC_score": 0.738
        },
        "GenAttack": {
            "accuracy": 0.025,
            "attack_success_rate": 0.882,
            "AUC_score": 0.027
        }
    },
    "FGSM-AT + entropy (eps=2/255)": {
        "No_attack": {"accuracy": 0.911, "AUC_score": 0.97},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.004,
            "attack_success_rate": 0.907,
            "AUC_score": 0.004
        },
        "NES": {
            "accuracy": 0.5,
            "attack_success_rate": 0.477,
            "AUC_score": 0.425
        },
        "GenAttack": {
            "accuracy": 0.0,
            "attack_success_rate": 0.911,
            "AUC_score": 0.0
        }
    },
    "FGSM-AT + entropy (eps=8/255)": {
        "No_attack": {"accuracy": 0.913, "AUC_score": 0.968},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.037,
            "attack_success_rate": 0.876,
            "AUC_score": 0.029
        },
        "NES": {
            "accuracy": 0.722,
            "attack_success_rate": 0.235,
            "AUC_score": 0.799
        },
        "GenAttack": {
            "accuracy": 0.03,
            "attack_success_rate": 0.883,
            "AUC_score": 0.044
        }
    },
    "FGSM-AT (cosine epsilon scheduler)": {
        "No_attack": {"accuracy": 0.86, "AUC_score": 0.93},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.091,
            "attack_success_rate": 0.774,
            "AUC_score": 0.104
        },
        "GenAttack": {
            "accuracy": 0.321,
            "attack_success_rate": 0.546,
            "AUC_score": 0.506
        },
        "NES": {
            "accuracy": 0.526,
            "attack_success_rate": 0.483,
            "AUC_score": 0.841
        }
    },
    "FGSM-AT (linear epsilon scheduler)": {
        "No_attack": {"accuracy": 0.782, "AUC_score": 0.87},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.053,
            "attack_success_rate": 0.729,
            "AUC_score": 0.063
        },
        "GenAttack": {
            "accuracy": 0.477,
            "attack_success_rate": 0.305,
            "AUC_score": 0.606
        },
        "NES": {
            "accuracy": 0.766,
            "attack_success_rate": 0.03,
            "AUC_score": 0.86
        }
    },
    "FGSM-AT + entropy (cosine epsilon scheduler)": {
        "No_attack": {"accuracy": 0.82, "AUC_score": 0.89},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.021,
            "attack_success_rate": 0.807,
            "AUC_score": 0.021
        },
        "GenAttack": {
            "accuracy": 0.392,
            "attack_success_rate": 0.436,
            "AUC_score": 0.5
        },
        "NES": {
            "accuracy": 0.703,
            "attack_success_rate": 0.255,
            "AUC_score": 0.824
        }
    },
    "FGSM-AT + entropy (linear epsilon scheduler)": {
        "No_attack": {"accuracy": 0.782, "AUC_score": 0.87},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.031,
            "attack_success_rate": 0.781,
            "AUC_score": 0.041
        },
        "GenAttack": {
            "accuracy": 0.532,
            "attack_success_rate": 0.28,
            "AUC_score": 0.633
        },
         "NES": {
            "accuracy": 0.792,
            "attack_success_rate": 0.034,
            "AUC_score": 0.867
        }
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
        "SQUARE (1000 iterations, eps=16/255)": {
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
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.072,
            "attack_success_rate": 0.741,
            "AUC_score": 0.076
        }
    }
}

data_adaptive = {
    "Clean_Model": {
        "No_attack": {"accuracy": 0.902, "AUC_score": 0.964},
        "FGSM (eps=2/255)": {"accuracy": 0.068, "attack_success_rate": 0.834, "AUC_score": 0.02},
        "FGSM (eps=4/255)": {"accuracy": 0.081, "attack_success_rate": 0.821, "AUC_score": 0.02},
        "FGSM (eps=8/255)": {"accuracy": 0.151, "attack_success_rate": 0.763, "AUC_score": 0.068},
        "IFGSM": {"accuracy": 0.143, "attack_success_rate": 0.759, "AUC_score": 0.068},
        "PGD": {"accuracy": 0.013, "attack_success_rate": 0.889, "AUC_score": 0.002},
        "JSMA": {"accuracy": 0.0, "attack_success_rate": 0.902, "AUC_score": 0.0}
    },
    "FGSM-AT + entropy (3 epochs cosine scheduler)": {
        "No_attack": {"accuracy": 0.82, "AUC_score": 0.91},
        "FGSM (eps=2/255)": {
            "accuracy": 0.531,
            "attack_success_rate": 0.301,
            "AUC_score": 0.38
        },
       "FGSM (eps=4/255)": {
            "accuracy": 0.413,
            "attack_success_rate": 0.429,
            "AUC_score": 0.206
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.349,
            "attack_success_rate": 0.509,
            "AUC_score": 0.133
        },
        "IFGSM": {
            "accuracy": 0.662,
            "attack_success_rate": 0.166,
            "AUC_score": 0.669
        },
        "PGD": {
            "accuracy": 0.531,
            "attack_success_rate": 0.297,
            "AUC_score": 0.35
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.906, "AUC_score": 0.0},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.071,
            "attack_success_rate": 0.757,
            "AUC_score": 0.068
        },
        "GenAttack": {
            "accuracy": 0.566,
            "attack_success_rate": 0.262,
            "AUC_score": 0.537
        },
        "NES": {
            "accuracy": 0.794,
            "attack_success_rate": 0.042,
            "AUC_score": 0.906
        }
    },
    "FGSM-AT + entropy (3 epochs linear scheduler)": {
        "No_attack": {"accuracy": 0.83, "AUC_score": 0.91},
         "FGSM (eps=2/255)": {
            "accuracy": 0.515,
            "attack_success_rate": 0.318,
            "AUC_score": 0.443
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.346,
            "attack_success_rate": 0.487,
            "AUC_score": 0.222
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.211,
            "attack_success_rate": 0.622,
            "AUC_score": 0.084
        },
        "IFGSM": {
            "accuracy": 0.693,
            "attack_success_rate": 0.14,
            "AUC_score": 0.732
        },
        "PGD": {
            "accuracy": 0.554,
            "attack_success_rate": 0.279,
            "AUC_score": 0.503
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.907, "AUC_score": 0.0},
        "GenAttack": {
            "accuracy": 0.594,
            "attack_success_rate": 0.239,
            "AUC_score": 0.607
        }

    },
    "FGSM-AT (3 epochs cosine scheduler)": {
        "No_attack": {"accuracy": 0.8, "AUC_score": 0.88},
        "FGSM (eps=2/255)": {
            "accuracy": 0.54,
            "attack_success_rate": 0.284,
            "AUC_score": 0.383
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.407,
            "attack_success_rate": 0.425,
            "AUC_score": 0.206
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.336,
            "attack_success_rate": 0.514,
            "AUC_score": 0.126
        },
        "IFGSM": {
            "accuracy": 0.673,
            "attack_success_rate": 0.147,
            "AUC_score": 0.665
        },
        "PGD": {
            "accuracy": 0.54,
            "attack_success_rate": 0.28,
            "AUC_score": 0.352
        },
        "JSMA": { "accuracy": 0.001, "attack_success_rate": 0.819, "AUC_score": 0.002},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.067,
            "attack_success_rate": 0.753,
            "AUC_score": 0.06
        }
    },
    "FGSM-AT (3 epochs linear scheduler)": {
        "No_attack": {"accuracy": 0.782, "AUC_score": 0.87},
        "FGSM (eps=2/255)": {
            "accuracy": 0.508,
            "attack_success_rate": 0.316,
            "AUC_score": 0.449
        },
        "FGSM (eps=4/255)": {
            "accuracy": 0.376,
            "attack_success_rate": 0.458,
            "AUC_score": 0.249
        },
        "FGSM (eps=8/255)": {
            "accuracy": 0.251,
            "attack_success_rate": 0.587,
            "AUC_score": 0.111
        },
        "IFGSM": {
            "accuracy": 0.681,
            "attack_success_rate": 0.143,
            "AUC_score": 0.727
        },
        "PGD": {
            "accuracy": 0.54,
            "attack_success_rate": 0.28,
            "AUC_score": 0.352
        },
        "JSMA": { "accuracy": 0.0, "attack_success_rate": 0.911, "AUC_score": 0.0},
        "SQUARE (1000 iterations, eps=16/255)": {
            "accuracy": 0.067,
            "attack_success_rate": 0.753,
            "AUC_score": 0.06
        }
    },
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
        "ZOO": {"avg_l2": 38.79, "avg_linf": 0.86},
        "AutoZOOM": {"avg_l2": 9.92, "avg_linf": 0.06}
}

df_l = pd.DataFrame(data_l).T.reset_index()
df_l.rename(columns={"index": "attack"}, inplace=True)
barplot_perturbation(df_l)

# Plot ASR
#plot_metric_bar(data_old, metric="attack_success_rate", log_scale=False, save_path="comparison_images/ASR_comparison_by_attack_b.png")
### Plot accuracy
#plot_metric_bar(data_old, metric="accuracy", log_scale=False, save_path="comparison_images/accuracy_comparison_by_attack_b.png")
### AUC
#plot_metric_bar(data_old, metric="AUC_score", log_scale=False, save_path="comparison_images/AUC_by_attack_b.png")

#plot_model_metrics_heatmap(
#    data_black,
#    metrics=["accuracy", "attack_success_rate", "AUC_score"],
#    output_dir="comparison_images",
#    attack_type="black"
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
