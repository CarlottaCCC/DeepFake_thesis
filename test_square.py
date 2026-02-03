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


def test_square(model, test_loader, device):

    results = {}

    clean_metrics = Metrics()
    square_metrics = Metrics()

    # Initializing classifier for square attack
    classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters()),
    input_shape=(3, 224, 224),
    nb_classes=2,
    clip_values=(0.0, 1.0),
    device_type="gpu"
    )

    attack = SquareAttack(
    estimator=classifier,
    norm="inf",     
    eps=8/255,
    max_iter=SQUARE_ITER,
    p_init=0.8
    )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    pbar = tqdm(test_loader, desc=f"Testing", unit="batch")

    for batch in pbar:
        if batch is None:
            continue
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device).long().squeeze()
        #print(y.shape, y.dtype)

        # Square Attack (L inf)
        imgs_square = attack.generate(x=imgs.numpy(), y=labels.numpy())
        ##ART returns numpy, need to convert imgs_square to tensor to pass it to the model
        imgs_square = torch.from_numpy(imgs_square).float().to(device)
        ##I normalize the images here, because Square needs not normalized images
        imgs_square = normalize(imgs_square)
        imgs = normalize(imgs)

        # inferenza
        with torch.no_grad():
            logits_clean = model(imgs)
            logits_square = model(imgs_square)

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            clean_metrics.update(labels, probs_clean)
            probs_square = torch.softmax(logits_square, dim=1)[:,1].detach().cpu().numpy()
            square_metrics.update(labels, probs_square)

    #Attack success rate
    square_metrics.attack_success_rate(clean_metrics.all_probs)
    
    clean_results = clean_metrics.compute()
    square_results = square_metrics.compute()

    print("CLEAN RESULTS")
    clean_metrics.print(0)
    print("SQUARE ATTACK RESULTS (black-box)")
    square_metrics.print(0)
    print(f"Attack Success Rate:  {square_metrics.asr_list[0]}")

    #saving metrics history
    history = {
        "clean_auc": clean_metrics.auc_list,
        "square_auc": square_metrics.auc_list,
        "clean_auc": clean_metrics.auc_list,
        "square_auc": square_metrics.auc_list,
        "clean_f1": clean_metrics.f1_list,
        "square_f1": square_metrics.f1_list,
        "clean_precision": clean_metrics.precision_list,
        "square_precision": square_metrics.precision_list,
        "clean_recall": clean_metrics.recall_list,
        "square_recall": square_metrics.recall_list,
        "clean_accuracy": clean_metrics.accuracy_list,
        "square_accuracy": square_metrics.accuracy_list,
        "square_asr": square_metrics.asr_list,
        "square_epsilon": TEST_EPS_SQUARE
    }
    save_history_json(history,f"test_results_SQUARE/trained_robust_on_SQUARE_results/results_robust_iter_{SQUARE_ITER}_EPS_{TEST_EPS_SQUARE}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}.json")

    return clean_metrics, square_metrics

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=None)
    # I modify the last layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # I load the trained clean model
    checkpoint_path = "models/clean_resnet50/resnet50_clean_epoch_20_LR_0.0003_batchsize_32_WD_1e-05.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
        
    print("Initializing testing dataset....")
    test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
    # I get a smaller subset of 500 images
    test_small = balanced_subset(test_dataset, n_per_class=32)
    
    print("Initializing test loader...")
    test_loader = DataLoader(test_small, batch_size=BATCH_SIZE, shuffle=True)

    clean_metrics, square_metrics = test_square(model, test_loader, device)

    # I save the results in a csv file
    #file_exists = os.path.isfile("test_results/trained_clean_results/test_epoch_16_LR_0.0003_batchsize_32_WD_1e-05_2.csv")

    #results = [
    #    ["Clean", 0, clean_metrics.accuracy_list[0], clean_metrics.precision_list[0], clean_metrics.recall_list[0], clean_metrics.f1_list[0], clean_metrics.auc_list[0], "none"],
    #    ["FGSM", EPS, fgsm_metrics.accuracy_list[0], fgsm_metrics.precision_list[0], fgsm_metrics.recall_list[0], fgsm_metrics.f1_list[0], fgsm_metrics.auc_list[0], fgsm_metrics.asr]
    #]
#
    ##["Square", EPS, square_metrics.accuracy_list[0], square_metrics.precision_list[0], square_metrics.recall_list[0], square_metrics.f1_list[0], square_metrics.auc_list[0], square_metrics.asr]
#
    #with open("test_results/trained_clean_results/test_epoch_16_LR_0.0003_batchsize_32_WD_1e-05_2.csv", "a", newline="") as f:
    #    writer = csv.writer(f)
    #
    #    if not file_exists:
    #        writer.writerow([
    #            "attack", "epsilon",
    #            "accuracy", "precision",
    #            "recall", "f1_score", "auc_score", "attack_success_rate"
    #        ])
    #
    #    writer.writerow(results)
    

    # plot roc curve
    plot_roc(clean_metrics.fpr, clean_metrics.tpr, clean_metrics.auc_list[0], "(test)", "clean_test")
    plot_roc(square_metrics.fpr, square_metrics.tpr, square_metrics.auc_list[0], "(test)", f"square_test_{SQUARE_ITER}_iterations")


