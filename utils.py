from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os
import torch
import random
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from torch.utils.data import Subset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
#CONTAINER COMMANDS

# models path: /mnt/hdd1/ciani/models/server_models:/models/
# dataset path: /mnt/hdd1/ciani/faceforensics:/data/
# code path: /home/ciani/tesi_server_deepfake:/work/project/
'''
podman run -it -v /home/ciani/tesi_server/deepfake:/work/project/ -v /mnt/hdd1/ciani/faceforensics:/data/ -v /mnt/hdd1/ciani/models/server_models:/models/  --hooks-dir=/usr/share/containers/oci/hooks.d/ --device nvidia.com/gpu=all  --ipc host localhost/ciani1881291/ciani_cirillo:latest
'''

'''
tmux session
tmux ls
tmux attach -t ID
'''

#### CONSTANTS ########
#ROOT_DIR = r"faceforensics"
ROOT_DIR = r"/data"
MODELS_DIR = r"/models"
MANIPULATION = "Deepfakes"
BATCH_SIZE = 32
LR = 1e-4
WD = 1e-2
NUM_EPOCHS = 30
DROPOUT = 0.0
LABEL_SMOOTHING = 0.0

EPS = 4/255
TEST_EPS_FGSM = 8/255
TEST_EPS_SQUARE = 16/255
SQUARE_ITER = 1000
LAMBDA_ENTROPY = 0.1
LAMBDA_ALIGN = 0.05

# ********* WHITE BOX ATTACKS *********
# PGD
EPS_PGD = 8/255
#IFGSM
EPS_IFGSM = 8/255

transform_size = transforms.Compose([
transforms.Resize((224, 224))
])

transform_jsma = transforms.Compose([
transforms.Resize((64, 64))
])


# function that counts the number of fake and real samples

def count_labels(dataset):
    labels = []
    for _, label in dataset:
        labels.append(int(label))
    counter = Counter(labels)
    return counter

# Class for computing metrics
class Metrics:
    def __init__(self):
        self.y_true = 0
        self.y_pred = 0
        self.y_prob = 0
        self.fpr = 0
        self.tpr = 0
        self.total_l2 = 0
        self.total_linf = 0
        self.avg_l2 = 0
        self.avg_linf = 0
        self.train_losses = []
        self.accuracy_list = []
        self.precision_list = []
        self.recall_list = []
        self.auc_list = []
        self.f1_list = []
        self.all_probs = []
        self.all_labels = []
        self.asr_list = []
        self.history = {}
    
    def reset_epoch(self):
        self.all_probs = []
        self.all_labels = []

    def update(self, labels, probs):
        """
        labels: torch.Tensor (B,1) o (B,)
        probs: numpy array (B,)
        """
        self.all_probs.append(probs)
        self.all_labels.append(labels.detach().cpu().numpy().ravel())

    def compute(self):
        y_true = np.concatenate(self.all_labels)
        y_prob = np.concatenate(self.all_probs)
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        #precision, recall and f1 computed on the FAKE class
        f1 = f1_score(y_true, y_pred, pos_label=1)
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        self.accuracy_list.append(acc)
        self.f1_list.append(f1)
        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.auc_list.append(roc_auc)
        self.fpr = fpr
        self.tpr = tpr

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": roc_auc
        }

    def attack_success_rate(self, all_probs_clean, threshold=0.5):
        y_true = np.concatenate(self.all_labels)
        y_true = np.asarray(y_true)
        probs_adv = np.concatenate(self.all_probs)
        probs_clean = np.concatenate(all_probs_clean)
    
        pred_clean = (probs_clean > threshold).astype(int)
        pred_adv   = (probs_adv > threshold).astype(int)
    
        # correct clean samples
        correct_clean = (pred_clean == y_true)
    
        # successful attacks - the attack changes the prediction (in general)
        #successful_attacks = correct_clean & (pred_adv != y_true)
        successful_attacks = pred_adv != pred_clean
    
        if correct_clean.sum() == 0:
            return 0.0  # no division by 0
        
        #asr = successful_attacks.sum() / correct_clean.sum()
        asr = successful_attacks.sum() / len(pred_adv)
        self.asr_list.append(asr)

    
    def print(self, epoch):
        print(f"Accuracy:  {self.accuracy_list[epoch]:.4f}")
        print(f"F1 score:   {self.f1_list[epoch]:.4f}")
        print(f"Precision:   {self.precision_list[epoch]:.4f}")
        print(f"Recall:   {self.recall_list[epoch]:.4f}")
        print(f"AUC score:   {self.auc_list[epoch]:.4f}")

def get_data_loaders(transform, batch_size):
    print("Initializing training dataset....")
    train_dataset = FFDataset(root_dir=ROOT_DIR, split="train", transform=transform)
    # I get a small subset for debugging
    #train_small, _ = balanced_subset(train_dataset, n_per_class=30)
    
    #print(train_dataset.getitem(0))
    print("Initializing validation dataset....")
    val_dataset = FFDataset(root_dir=ROOT_DIR, split="val", transform=transform)
    print("Initializing testing dataset....")
    test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
    #with open("sampled_test_set.json") as f:
    #    sampled_test_set_paths = json.loads(f.read())
    #test_small = get_imgs_by_filepath(test_dataset, sampled_test_set_paths)
    test_small, _ = balanced_subset(test_dataset, n_per_class=500)
    
    print("Initializing train loader...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Initializing val loader....")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Initializing test loader....")
    test_loader = DataLoader(test_small, batch_size=batch_size, shuffle=False)    
    return train_loader, val_loader, test_loader


def entropy_penalty(logits, eps=1e-8):
    #logits: (B, C)
    #returns: scalar entropy (mean over batch)
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=1)  # (B,)
    return entropy.mean()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_acc = None
        self.best_epoch = 0
        self.best_epoch = 0
        self.best_eps = 0
        self.best_weights = None  # in-memory backup (no disk needed)

    def __call__(self, val_acc, model, optimizer, epoch, seed, eps_sched, attack, train_metrics_clean, train_metrics_adv, val_metrics, train_losses, current_eps):
        if self.best_acc is None or val_acc < self.best_acc - self.min_delta:
            self.best_acc = val_acc
            self.best_epoch = epoch
            #self.best_eps = current_eps
            save_path =  f'{MODELS_DIR}/resnet50_clean_epoch_{epoch+1}_with_Multistep_lrscheduler.pt'
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            #torch.save(model.state_dict(), save_path)  # save the best
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'train_losses': train_losses,
                "train_auc_clean": train_metrics_clean.auc_list,
                #"train_auc_adv": None,
                "val_auc": val_metrics.auc_list,
                "train_tpr_clean": train_metrics_clean.tpr,
                #"train_tpr_adv": None,
                "val_tpr": val_metrics.tpr,
                "train_fpr_clean": train_metrics_clean.fpr,
                #"train_fpr_adv": None,
                "val_fpr": val_metrics.fpr
            },save_path)

            # save history
            history = {
                "epoch": self.best_epoch+1,
                "train_losses": train_losses,
                "train_auc_clean": train_metrics_clean.auc_list,
                #"train_auc_adv": train_metrics_adv.auc_list,
                "val_auc": val_metrics.auc_list,
                "train_f1_clean": train_metrics_clean.f1_list,
                #"train_f1_adv": train_metrics_adv.f1_list,
                "val_f1": val_metrics.f1_list,
                "train_precision_clean": train_metrics_clean.precision_list,
                #"train_precision_adv": train_metrics_adv.precision_list,
                "val_precision": val_metrics.precision_list,
                #"train_recall_clean": train_metrics_clean.recall_list,
                #"train_recall_adv": train_metrics_adv.recall_list,
                "val_recall": val_metrics.recall_list,
                "train_accuracy_clean": train_metrics_clean.accuracy_list,
                #"train_accuracy_adv": train_metrics_adv.accuracy_list,
                "val_accuracy": val_metrics.accuracy_list,
                #"train_asr": train_metrics_adv.asr_list,
                #"train_epsilon": self.best_eps
            }
            save_history_json(history,f"history/history_clean/history_resnet50_clean_epoch_{epoch+1}_with_Multistep_lrscheduler.json")
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping!")
                return True
        return False
    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"Restored best model (val_loss={self.best_loss:.4f})")

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EpsilonScheduler:
    def __init__(self, eps_start=0/255, eps_end=8/255, num_epochs_rampup=10, type='linear'):
        self.eps_start        = eps_start
        self.eps_end          = eps_end
        self.num_epochs_rampup = num_epochs_rampup
        self.type = type
        self.num_epochs_per_eps=num_epochs_rampup

    def get_epsilon(self, epoch):
        t = min(epoch / self.num_epochs_rampup, 1.0)  # clamp to [0,1]
        
        # Linear: slow and steady
        linear = self.eps_start + (self.eps_end - self.eps_start) * t
    
        # Cosine: gentler start, faster finish
        cosine = self.eps_start + (self.eps_end - self.eps_start) * (1 - math.cos(math.pi * t)) / 2
    
        # Exponential: very gentle start
        exponential = self.eps_start + (self.eps_end - self.eps_start) * (t ** 2)

        if self.type == 'linear':
            return linear
        elif self.type == 'cosine':
            return cosine
        elif self.type == 'exponential':
            return exponential

class CurriculumEpsilonScheduler:
    def __init__(
        self,
        eps_start: float = 0 / 255,
        eps_end: float = 8 / 255,
        num_epochs_rampup: int = 10,
        type: str = 'linear',
        epsilons=[], #list of epsilons
        epochs_per_epsilon=[], #list of epochs, epsilons[i] will have epochs_per_epsilon[i]
        adaptive: bool = False,
        patience: int = 5,
        min_delta: float = 1e-4,
        min_clean_acc: float = 0.85,
        num_epochs_per_eps=3
    ):
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.num_epochs_rampup = num_epochs_rampup
        self.type = type
        self.adaptive = adaptive
        self.patience = patience
        self.min_delta = min_delta
        self.min_clean_acc = min_clean_acc
        self.num_epochs_per_eps = num_epochs_per_eps

        if self.type == 'random':
            self.epsilons = self.get_random_epsilon(num_epochs=num_epochs_rampup,eps_max_pixel=8)
            print(f"[CurriculumScheduler] Random schedule:")
            for i, eps in enumerate(self.epsilons):
                print(f"  Step {i+1}/{num_epochs_rampup}: ε = {eps:.5f}")
        else:
            # Pre-compute all epsilon steps from the schedule
            self.epsilons = [
                self.compute_epsilon(i)
                for i in range(0, num_epochs_rampup)
            ]
            print(f"[CurriculumScheduler] Adaptive schedule ({type}):")
            for i, eps in enumerate(self.epsilons):
                print(f"  Step {i+1}/{num_epochs_rampup}: ε = {eps:.5f}")
        #else:
        #    assert epsilons is not None and epochs_per_epsilon is not None, \
        #        "Non-adaptive mode requires explicit `epsilons` and `epochs_per_epsilon` lists."
        #    assert len(epsilons) == len(epochs_per_epsilon), \
        #        "epsilons and epochs_per_epsilon must have the same length."
        #    self.epsilons = epsilons
        #    self.epochs_per_epsilon = epochs_per_epsilon
        #    print(f"[CurriculumScheduler] Fixed schedule:")
        #    for i, (eps, ep) in enumerate(zip(self.epsilons, self.epochs_per_epsilon)):
        #        print(f"  Step {i+1}/{len(epsilons)}: ε = {eps:.5f}  ({ep} epochs)")

        self.best_acc = 0.0
        self.eps_counter = 0
        self.patience_counter = 0
        self.current_epsilon = 0
        self.epoch_epsilon_counter = 0
        self.current_epsilon_index = 0
        self.stopped=False


    def get_random_epsilon(self, num_epochs=10, eps_max_pixel=8):
        # Generate one epsilon per pixel value, then shuffle
        epsilons = [i / 255.0 for i in range(eps_max_pixel + 1)]  # [0, 1/255, ..., 8/255]
        
        # If num_epochs > num_levels, repeat and shuffle
        full = (epsilons * (num_epochs // len(epsilons) + 1))[:num_epochs]
        random.shuffle(full)
        return full

    def cyclic_epsilon(self, epoch, num_epochs, eps_max=8/255, num_cycles=3):
        cycle_length = num_epochs / num_cycles
        cycle_pos    = (epoch % cycle_length) / cycle_length
        return eps_max * 0.5 * (1 - math.cos(math.pi * cycle_pos))
 

    def compute_epsilon(self, counter:int):
        t = min(counter / self.num_epochs_rampup, 1.0)  # clamp to [0,1]
        
        # Linear: slow and steady
        linear = self.eps_start + (self.eps_end - self.eps_start) * t
    
        # Cosine: gentler start, faster finish
        cosine = self.eps_start + (self.eps_end - self.eps_start) * (1 - math.cos(math.pi * t)) / 2
    
        # Exponential: very gentle start
        exponential = self.eps_start + (self.eps_end - self.eps_start) * (t ** 2)

        cyclic = self.cyclic_epsilon(counter, self.num_epochs_rampup)

        if self.type == 'linear':
            return linear
        elif self.type == 'cosine':
            return cosine
        elif self.type == 'exponential':
            return exponential
        elif self.type == 'random':
            return self.epsilons[counter]
        elif self.type == 'cyclic':
            return cyclic
        elif self.type == 'None':
            return 'None'
        else:
            raise ValueError(f"Unknown scheduler type '{self.type}'")
        
    
    def get_epsilon(self, val_acc_adv: float, clean_acc: float = None, epoch = 0):

        # clean accuracy check for catastrofic forgetting
        #if clean_acc is not None and clean_acc < self.min_clean_acc and epoch > 0:
        #    print(f"[WARNING] Clean acc {clean_acc:.3f} below threshold "
        #          f"{self.min_clean_acc:.3f} — possible catastrophic forgetting.")
        #    self.stopped = True
        #    return self.epsilons[self.current_epsilon_index]

        # All steps exhausted
        if self.current_epsilon_index >= len(self.epsilons):
            self.stopped = True
            return self.epsilons[-1]
       
        # I train with a list of epsilons that I want 
        # with the number of epochs that I want
        #self.epoch_epsilon_counter += 1
        #print(f"epsilon index: {self.current_epsilon_index}")
        #print(f"epoch counter {self.epoch_epsilon_counter}")
        if self.epoch_epsilon_counter >= self.num_epochs_per_eps:
            old_eps = self.epsilons[self.current_epsilon_index]
            self.current_epsilon_index += 1
            self.epoch_epsilon_counter = 1
            if self.current_epsilon_index < len(self.epsilons):
                new_eps = self.epsilons[self.current_epsilon_index]
                #print(f"[CurriculumScheduler] Budget reached → advance: "
                #      f"ε {old_eps:.5f} → {new_eps:.5f}")
            else:
                print(f"[CurriculumScheduler] All steps complete.")
                self.stopped = True
                return self.epsilons[-1]
            
        else:
            self.epoch_epsilon_counter += 1
        
        
        return self.epsilons[self.current_epsilon_index]
    
    
    def should_stop(self) -> bool:
        return self.stopped
        


def plot_roc(fpr, tpr, auc_score, epoch, title):
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Validation ROC Curve - Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.savefig(title, dpi=300)



def plot_metric(train_list, val_list, num_epochs, metric_name, title):
    epochs = range(1, num_epochs + 1)

    plt.figure()
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in train_list], label=f"Train {metric_name}")
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in val_list], label=f"Val {metric_name}")
    plt.title(f"{metric_name} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(title, dpi=300)
    plt.close()


def plot_loss(train_losses, path):
    epochs = range(1, len(train_losses)+1)
    plt.figure()
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in train_losses], label="Train Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300)
    plt.close()

def plot_model_metrics_heatmap(
    data,
    metrics,
    output_dir="plots",
    figsize=(12, 6),
    dpi=300,
    attack_type='white',
    cmap='RdYlGn'
):
    """
    data: dict -> model -> attack -> metric -> value
    metrics: list of metric names (e.g. ["accuracy", "attack_success_rate", "AUC_score"])
    """
    models = list(data.keys())
    attacks = list(next(iter(data.values())).keys())
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        # Costruisce la matrice (modelli x attacchi)
        matrix = []
        for model in models:
            row = []
            for attack in attacks:
                v = data[model][attack].get(metric, np.nan)
                row.append(np.nan if v is None else float(v))
            matrix.append(row)
        matrix = np.array(matrix)

        fig, ax = plt.subplots(figsize=figsize)

        # Per attack_success_rate inverti la colormap (alto = peggio)
        current_cmap = cmap + "_r" if metric == "attack_success_rate" else cmap

        im = ax.imshow(matrix, aspect='auto', cmap=current_cmap, vmin=0, vmax=1)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric, fontsize=11)

        # Assi
        ax.set_xticks(np.arange(len(attacks)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(attacks, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(models, fontsize=9)

        # Annotazioni: valore numerico in ogni cella
        for i in range(len(models)):
            for j in range(len(attacks)):
                val = matrix[i, j]
                if not np.isnan(val):
                    # Testo bianco o nero a seconda del valore di sfondo
                    brightness = val if metric != "attack_success_rate" else 1 - val
                    text_color = "white" if brightness < 0.4 or brightness > 0.75 else "black"
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            fontsize=8, color=text_color, fontweight='bold')
                else:
                    ax.text(j, i, "N/A", ha='center', va='center',
                            fontsize=8, color='gray')

        ax.set_title(f"Models Comparison — {metric}", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Attack", fontsize=11)
        ax.set_ylabel("Model", fontsize=11)

        plt.tight_layout()
        filename = f"{metric}_heatmap_{attack_type}_2.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches="tight")
        plt.close()

def plot_metric_bar(data_dict, metric, log_scale=False, save_path=None):
    """
    Crea un bar plot comparativo per una metrica specifica dai risultati dei modelli.

    Parameters:
    - data_dict: dict, struttura tipo data_old
    - metric: str, metrica da plottare (es. 'attack_success_rate', 'accuracy', 'AUC_score')
    - log_scale: bool, se True imposta scala logaritmica sull'asse y
    - save_path: str, path per salvare il plot (default None)
    """
    
    # Costruisci DataFrame
    rows = []
    for model, attacks in data_dict.items():
        for attack, metrics in attacks.items():
            value = metrics.get(metric, None)
            if value is not None:
                rows.append({"model": model, "attack": attack, metric: value})
    
    df = pd.DataFrame(rows)
    
    # Plot
    plt.figure(figsize=(12,6))
    palette = sns.color_palette("Set2", n_colors=df['model'].nunique())
    ax = sns.barplot(data=df, x="attack", y=metric, hue="model", palette=palette, edgecolor="black")
    
    # Scala logaritmica se richiesta
    if log_scale:
        ax.set_yscale('log')
    
    # Annotazioni sopra le barre
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, rotation=0) # to put the numbers horizontally rotation=0
    
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} per attack on different models")
    plt.legend(title="Model")
    plt.tight_layout()
    
    # Salvataggio
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
    

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def save_results_for_data(
    model_name: str,
    attack_label: str,
    accuracy: float,
    auc: float = None,
    asr: float = None,
    avg_l2: float = None,
    avg_linf: float = None,
    output_path: str = "results_for_data.json"
):
    """
    Salva i risultati nel formato pronto per essere copiato in 'data'.
    
    Parametri
    ---------
    model_name   : es. "FGSM-AT (eps=8/255)"
    attack_label : es. "PGD (eps=8/255)"
    accuracy     : valore float
    auc          : valore float, None se non disponibile
    asr          : valore float, None se non disponibile (non presente per "No attack")
    output_path  : path del file JSON di output
    """
    # carica il file esistente se c'è, altrimenti parte da zero
    try:
        with open(output_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    # crea la struttura per il modello se non esiste
    if model_name not in data:
        data[model_name] = {}

    # costruisce il dict del risultato
    result = {"accuracy": round(accuracy, 3)}
    if asr is not None:
        result["attack_success_rate"] = round(asr, 3)
    if auc is not None:
        result["AUC_score"] = round(auc, 3)

    data[model_name][attack_label] = result

    # salva
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    # stampa la riga pronta per data
    print(f'\n"{attack_label}": {json.dumps(result)}\n')

def balanced_subset(dataset, n_per_class=36, seed=42):
    #random.seed(seed)

    real_idxs = [i for i, (_, y, _) in enumerate(dataset.samples) if y == 0]
    fake_idxs = [i for i, (_, y, _) in enumerate(dataset.samples) if y == 1]

    real_sel = random.sample(real_idxs, n_per_class)
    fake_sel = random.sample(fake_idxs, n_per_class)

    indices = real_sel + fake_sel
    random.shuffle(indices)

    # now we return also the img_ids
    img_ids = [dataset.samples[i][0] for i in indices]

    return Subset(dataset, indices), img_ids


def get_imgs_by_filepath(dataset, filepaths:list[str]):
    indices = []
    for i, elem in enumerate(dataset.samples):
        if elem[0] in filepaths:
            indices.append(i)
    return Subset(dataset, indices)


def save_history_json(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(history, f, indent=4)

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def plot_model_metrics_by_attack(
    data,
    metrics,
    output_dir="plots",
    figsize=(12, 6),
    bar_width=0.25,
    dpi=300
):
    """
    data: dict -> model -> attack -> metric -> value
    metrics: list of metric names (e.g. ["accuracy", "attack_success", "AUC_score"])
    """

    models = list(data.keys())
    attacks = list(next(iter(data.values())).keys())

    n_models = len(models)
    x = np.arange(len(attacks))

    os.makedirs(output_dir, exist_ok=True)

    MODEL_COLORS = {
        "Clean_Model": "#E69F00",   # arancio soft
        "FGSM-AT (epsilon=2/255)": "#4C78A8",    # azzurro soft
        "FGSM-AT (epsilon=4/255)": "#6B8FB3", # blue
        "FGSM-AT (epsilon=8/255)": "#9FBAD6", # blue
        "SQUARE_Model_1": "#009E73",  # verde soft
        "SQUARE_Model_2": "#009ED6"  # verde soft
    }

    for metric in metrics:
        plt.figure(figsize=figsize)

        for i, model in enumerate(models):
            values = [
                data[model][attack].get(metric, np.nan)
                for attack in attacks
            ]

            plt.bar(
                x + i * bar_width,
                values,
                width=bar_width,
                label=model,
                color=MODEL_COLORS.get(model, "gray")
            )

        plt.xlabel("Attack")
        plt.ylabel(metric)
        plt.title(f"Models Comparison on {metric}")

        plt.xticks(
            x + bar_width * (n_models - 1) / 2,
            attacks,
            rotation=30,
            ha="right"
        )

        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        filename = f"{metric}_by_attack_2.png"
        plt.savefig(
            os.path.join(output_dir, filename),
            dpi=dpi,
            bbox_inches="tight"
        )

        plt.close()


# function to compute the average L2 and L_inf norms
def batch_norms(delta):
    # flatten for each sample
    delta_flat = delta.view(delta.size(0), -1)

    l2 = torch.norm(delta_flat, p=2, dim=1)      # (B,)
    linf = torch.norm(delta_flat, p=float('inf'), dim=1)  # (B,)

    return l2, linf


def get_checkpoint(model, checkpoint_path, history_path, train_metrics, val_metrics, optimizer, device):
    # ricarico il checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] # riparte dall'epoch successivo
    print(f"Riprendo dal epoch {start_epoch}")
    train_losses = checkpoint["train_losses"]

    train_metrics.auc_list = checkpoint["train_auc"]
    val_metrics.auc_list = checkpoint["val_auc"]
    
    train_metrics.tpr = checkpoint["train tpr"]
    val_metrics.tpr = checkpoint["val tpr"]
    
    train_metrics.fpr = checkpoint["train fpr"]
    val_metrics.fpr = checkpoint["val fpr"]

    with open(history_path, "r") as f:
        history = json.load(f)
    
    train_metrics.f1_list = history["train_f1"]
    val_metrics.f1_list = history["val_f1"]
    
    train_metrics.precision_list = history["train_precision"]
    val_metrics.precision_list = history["val_precision"]
    
    train_metrics.recall_list = history["train_recall"]
    val_metrics.recall_list = history["val_recall"]
    
    train_metrics.accuracy_list = history["train_accuracy"]
    val_metrics.accuracy_list = history["val_accuracy"]

    return model, train_metrics, val_metrics, train_losses, start_epoch

def reset_checkpoint(num_epochs, checkpoint_path, optimizer_type, sched_type, device, base_lr=1e-7, max_lr=1e-6, wd=1e-2, step_size_up=1):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,              # your empirically-successful ceiling was ~1e-5 under CyclicLR;
                               # SGD+momentum can typically tolerate a somewhat higher peak LR
                               # than Adam-family optimizers for the same model, but I'd still
                               # start conservatively and treat this as a value to sweep
        momentum=0.9,
        weight_decay=wd,    # matches the paper directly, no scaling needed here
        )


    if sched_type == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=base_lr, 
        max_lr=max_lr, 
        step_size_up=step_size_up, 
        mode='triangular2' 
        )
    elif sched_type == "CosineAWR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=step_size_up,  # restart every epsilon level (in iterations)
        T_mult=1,
        eta_min=base_lr
    )
    elif sched_type == "MultiStepLR":
        # paper: decay at epochs 75/90 out of 100 → i.e. at 75% and 90% of total training
        milestones = [int(0.75 * num_epochs), int(0.90 * num_epochs)]  # e.g. [15, 18] for your 20 epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    return model, criterion, optimizer, scheduler

def reset_checkpoint_simple(checkpoint_path, device):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, 
    base_lr=1e-7, 
    max_lr=1e-4, 
    step_size_up=500, 
    mode='triangular2' 
    )

    return model, criterion, optimizer, scheduler

