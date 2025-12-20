from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

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
        self.accuracy_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_list = []
        self.auc_list = []
    
    def reset(self):
        self.y_true = 0
        self.y_pred = 0
        self.y_prob = 0

    def compute(self, y_true, y_pred, y_prob):
        self.y_true = y_true
        self.y_pred = y_pred

        acc = accuracy_score(y_true, y_pred)
        #precision, recall and f1 computed on the FAKE class
        f1 = f1_score(y_true, y_pred, pos_label=1)
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)

        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")

        self.accuracy_list.append(acc)
        self.f1_list.append(f1)
        self.auc_list.append(auc)
        self.precision_list.append(precision)
        self.recall_list.append(recall)
    
    def print(self, epoch):
        print(f"Accuracy:  {self.accuracy_list[epoch]:.4f}")
        print(f"F1 score:   {self.f1_list[epoch]:.4f}")
        print(f"Precision:   {self.precision_list[epoch]:.4f}")
        print(f"Recall:   {self.recall_list[epoch]:.4f}")
        print(f"AUC score:   {self.auc_list[epoch]:.4f}")

    #TODO
    #FUNZIONE PLOT METRICHE





