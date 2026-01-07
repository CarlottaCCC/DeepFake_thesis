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
import csv


def test(model, test_loader, device):

    clean_metrics = Metrics()
    fgsm_metrics = Metrics()
    square_metrics = Metrics()
    
    # I define the image bounds for the fmodel in order to properly attack in that space
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    lower = (0 - mean) / std
    upper = (1 - mean) / std
    model.eval()

    fmodel = fb.PyTorchModel(model, bounds=(lower.min().item(), upper.max().item()), device=device)
    pbar = tqdm(test_loader, desc=f"Testing", unit="batch")

    for batch in pbar:
        if batch is None:
            continue
        imgs, y = batch
        imgs, y = imgs.to(device), y.to(device).long().squeeze()
        print(y.shape, y.dtype)


        # FGSM
        fgsm = fb.attacks.FGSM()
        eps = EPS
        _, adv_fgsm, _ = fgsm(fmodel, imgs, y, epsilons=eps)

        # Square Attack (L∞)
        square = fb.attacks.SquareAttack(distance=fb.distances.linf)

        _, adv_square, _ = square(fmodel, imgs, y, epsilons=eps)

        # inferenza
        with torch.no_grad():
            logits_clean = model(imgs)
            logits_fgsm = model(adv_fgsm)
            print(logits_fgsm.shape)
            logits_square = model(adv_square)

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            clean_metrics.update(y, probs_clean)
            probs_fgsm = torch.softmax(logits_fgsm, dim=1)[:,1].detach().cpu().numpy()
            fgsm_metrics.update(y, probs_fgsm)
            probs_square = torch.softmax(logits_square, dim=1)[:,1].detach().cpu().numpy()
            square_metrics.update(y, probs_square)

    #Attack success rate
    fgsm_metrics.attack_success_rate(probs_clean, probs_fgsm)
    fgsm_metrics.attack_success_rate(probs_clean, probs_square)
    
    clean_results = clean_metrics.compute()
    fgsm_results = fgsm_metrics.compute()
    square_results = square_metrics.compute()

    print("CLEAN RESULTS")
    clean_metrics.print(0)
    print("FGSM ATTACK RESULTS (white-box)")
    clean_metrics.print(0)
    print(f"Attack Success Rate:  {fgsm_metrics.asr}")
    print("SQUARE ATTACK RESULTS (black-box)")
    clean_metrics.print(0)
    print(f"Attack Success Rate:  {square_metrics.asr}")

    return clean_metrics, fgsm_metrics, square_metrics

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=None)
    # I modify the last layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # I load the trained clean model
    checkpoint_path = "models/clean_resnet50/resnet50_clean_epoch_5_LR_0.0003_batchsize_16_WD_0.0001_2.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
       ])
        
    print("Initializing testing dataset....")
    test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
    
    print("Initializing test loader...")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    clean_metrics, fgsm_metrics, square_metrics = test(model, test_loader, device)

    # I save the results in a csv file
    file_exists = os.path.isfile("results.csv")

    results = [
        ["Clean", 0, clean_metrics.accuracy_list[0], clean_metrics.precision_list[0], clean_metrics.recall_list[0], clean_metrics.f1_list[0], clean_metrics.auc_list[0], "none"],
        ["FGSM", EPS, fgsm_metrics.accuracy_list[0], fgsm_metrics.precision_list[0], fgsm_metrics.recall_list[0], fgsm_metrics.f1_list[0], fgsm_metrics.auc_list[0], fgsm_metrics.asr],
        ["Square", EPS, square_metrics.accuracy_list[0], square_metrics.precision_list[0], square_metrics.recall_list[0], square_metrics.f1_list[0], square_metrics.auc_list[0], square_metrics.asr]
    ]

    with open("results.csv", "a", newline="") as f:
        writer = csv.writer(f)
    
        if not file_exists:
            writer.writerow([
                "attack", "epsilon",
                "accuracy", "precision",
                "recall", "f1_score", "auc_score", "attack_success_rate"
            ])
    
        writer.writerow(results)
    

    # plot roc curve
    plot_roc(clean_metrics.fpr, clean_metrics.tpr, clean_metrics.auc_list[0], "(test)")
    plot_roc(fgsm_metrics.fpr, fgsm_metrics.tpr, fgsm_metrics.auc_list[0], "(test)")
    plot_roc(square_metrics.fpr, square_metrics.tpr, square_metrics.auc_list[0], "(test)")


