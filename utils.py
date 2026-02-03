from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import torch
import random
from torch.utils.data import Subset

#### CONSTANTS ########
ROOT_DIR = r"faceforensics\data5"
BATCH_SIZE = 32
LR = 3e-4
WD = 1e-5
NUM_EPOCHS = 20

EPS = 2/255
TEST_EPS_FGSM = 2/255
TEST_EPS_SQUARE = 16/255
SQUARE_ITER = 5000
LAMBDA_ENTROPY = 0.01

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
    
        # successful attacks
        successful_attacks = correct_clean & (pred_adv != y_true)
    
        if correct_clean.sum() == 0:
            return 0.0  # no division by 0
        
        asr = successful_attacks.sum() / correct_clean.sum()
        self.asr_list.append(asr)

    
    def print(self, epoch):
        print(f"Accuracy:  {self.accuracy_list[epoch]:.4f}")
        print(f"F1 score:   {self.f1_list[epoch]:.4f}")
        print(f"Precision:   {self.precision_list[epoch]:.4f}")
        print(f"Recall:   {self.recall_list[epoch]:.4f}")
        print(f"AUC score:   {self.auc_list[epoch]:.4f}")



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


def plot_loss(train_losses):
    epochs = range(1, len(train_losses)+1)
    plt.figure()
    plt.plot(epochs, [l.detach().cpu().item() if torch.is_tensor(l) else l for l in train_losses], label="Train Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"metrics_images/loss_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}.png", dpi=300)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def balanced_subset(dataset, n_per_class=36, seed=42):
    random.seed(seed)

    real_idxs = [i for i, (_, y) in enumerate(dataset.samples) if y == 0]
    fake_idxs = [i for i, (_, y) in enumerate(dataset.samples) if y == 1]

    real_sel = random.sample(real_idxs, n_per_class)
    fake_sel = random.sample(fake_idxs, n_per_class)

    indices = real_sel + fake_sel
    random.shuffle(indices)

    return Subset(dataset, indices)

def save_history_json(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(history, f, indent=4)

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


import matplotlib.pyplot as plt
import numpy as np
import os

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
        "FGSM_Model": "#56B4E9",    # azzurro soft
        "SQUARE_Model": "#009E73",  # verde soft
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

        filename = f"{metric}_by_attack.png"
        plt.savefig(
            os.path.join(output_dir, filename),
            dpi=dpi,
            bbox_inches="tight"
        )

        plt.close()
