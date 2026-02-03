from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import foolbox as fb
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SquareAttack
import torch.optim as optim
import csv


def test_fgsm(model, test_loader, device):

    clean_metrics = Metrics()
    fgsm_metrics = Metrics()

    # I define the image bounds for the fmodel in order to properly attack in that space
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    lower = (0 - mean) / std
    upper = (1 - mean) / std
    model.eval()

    fmodel = fb.PyTorchModel(model, bounds=(lower.min().item(), upper.max().item()), device=device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    pbar = tqdm(test_loader, desc=f"Testing", unit="batch")

    for batch in pbar:
        if batch is None:
            continue
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device).long().squeeze()
        #print(y.shape, y.dtype)
        imgs = normalize(imgs)

        # FGSM
        fgsm = fb.attacks.FGSM()
        eps = TEST_EPS_FGSM
        _, imgs_fgsm, _ = fgsm(fmodel, imgs, labels, epsilons=eps)

        # inferenza
        with torch.no_grad():
            logits_clean = model(imgs)
            logits_fgsm = model(imgs_fgsm)

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            clean_metrics.update(labels, probs_clean)
            probs_fgsm = torch.softmax(logits_fgsm, dim=1)[:,1].detach().cpu().numpy()
            fgsm_metrics.update(labels, probs_fgsm)
    
    #Attack success rate
    fgsm_metrics.attack_success_rate(clean_metrics.all_probs)
   
    clean_results = clean_metrics.compute()
    fgsm_results = fgsm_metrics.compute()
    
    print("CLEAN RESULTS")
    clean_metrics.print(0)
    print("FGSM ATTACK RESULTS (white-box)")
    fgsm_metrics.print(0)
    print(f"Attack Success Rate:  {fgsm_metrics.asr_list[0]}")
   

    #saving metrics history
    history = {
        "clean_auc": clean_metrics.auc_list,
        "fgsm_auc": fgsm_metrics.auc_list,
        "clean_auc": clean_metrics.auc_list,
        "fgsm_auc": fgsm_metrics.auc_list,
        "clean_f1": clean_metrics.f1_list,
        "fgsm_f1": fgsm_metrics.f1_list,
        "clean_precision": clean_metrics.precision_list,
        "fgsm_precision": fgsm_metrics.precision_list,
        "clean_recall": clean_metrics.recall_list,
        "fgsm_recall": fgsm_metrics.recall_list,
        "clean_accuracy": clean_metrics.accuracy_list,
        "fgsm_accuracy": fgsm_metrics.accuracy_list,
        "fgsm_asr": fgsm_metrics.asr_list,
        "fgsm_epsilon_train": EPS,
        "fgsm_epsilon_test": TEST_EPS_FGSM
    }
    save_history_json(history,f"test_results_fgsm/trained_square_results/results_square_eps_{TEST_EPS_FGSM}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}.json")

    return clean_metrics, fgsm_metrics

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=None)
    # I modify the last layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # I load the trained clean model
    checkpoint_path = "models/square_resnet50/resnet50_square_epoch_10_LR_0.0003_batchsize_32_WD_1e-05.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
        
    print("Initializing testing dataset....")
    test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
    # I get a smaller subset of 500 images
    test_small = balanced_subset(test_dataset, n_per_class=250)
    
    print("Initializing test loader...")
    test_loader = DataLoader(test_small, batch_size=BATCH_SIZE, shuffle=True)

    clean_metrics, fgsm_metrics = test_fgsm(model, test_loader, device)

    # plot roc curve
    plot_roc(clean_metrics.fpr, clean_metrics.tpr, clean_metrics.auc_list[0], "(test)", "clean_test")
    plot_roc(fgsm_metrics.fpr, fgsm_metrics.tpr, fgsm_metrics.auc_list[0], "(test)", "fgsm_test")