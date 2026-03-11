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
from art.attacks.evasion import SquareAttack, ZooAttack, HopSkipJump
import torch.optim as optim
import csv

def test_zoo(model, test_loader, device):

    model.eval()

    results = {}

    clean_metrics = Metrics()
    zoo_metrics = Metrics()

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

    zoo = ZooAttack(
    classifier=classifier,
    confidence=0.0,           # Confidenza minima per adversarial example
    targeted=False,           # False per untargeted attack
    learning_rate=1e-2,       # Learning rate per l'ottimizzazione
    max_iter=20,             # Numero massimo di iterazioni
    binary_search_steps=10,   # Step per binary search
    initial_const=1e-3,       # Costante iniziale
    abort_early=True,         # Stop se attack ha successo
    use_resize=True,         # Resize dell'immagine (più veloce ma meno accurato)
    use_importance=True,      # Usa importance sampling
    nb_parallel=128,          # Numero di query parallele
    batch_size=1,
    variable_h=0.01,          # Step size per finite differences
    verbose=True
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

        # ZOO ATTTACK
        imgs_zoo = zoo.generate(x=imgs.cpu().numpy(), y=labels.cpu().numpy())
        imgs_zoo = torch.from_numpy(imgs_zoo).float().to(device)

        # normalize
        imgs_zoo = normalize(imgs_zoo)
        imgs = normalize(imgs)

        # inferenza
        with torch.no_grad():
            logits_clean = model(imgs)
            logits_zoo = model(imgs_zoo)

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            clean_metrics.update(labels, probs_clean)
            probs_zoo = torch.softmax(logits_zoo, dim=1)[:,1].detach().cpu().numpy()
            zoo_metrics.update(labels, probs_zoo)

    #Attack success 
    zoo_metrics.attack_success_rate(clean_metrics.all_probs)
    
    clean_results = clean_metrics.compute()
    zoo_results = zoo_metrics.compute()

    print("CLEAN RESULTS (50 images)")
    clean_metrics.print(0)

    print("ZOO ATTACK RESULTS (black-box) - (50 images)")
    zoo_metrics.print(0)
    print(f"Attack Success Rate:  {zoo_metrics.asr_list[0]}")

    return clean_metrics, zoo_metrics


def test_black(model, test_loader, model_name, device):

    model.eval()

    results = {}

    clean_metrics = Metrics()
    square_metrics = Metrics()
    nes_metrics = Metrics()

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

    # SQUARE ATTACK
    square = SquareAttack(
    estimator=classifier,
    norm="inf",     
    eps=8/255,
    max_iter=SQUARE_ITER,
    p_init=0.8
    )

    fmodel = fb.PyTorchModel(model, bounds=(0,1), device=device)

    attack = HopSkipJump(
    classifier=classifier,
    max_iter=50,
    max_eval=5000,
    init_eval=100,
    norm=np.inf  # Per L-infinity
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

        # SQUARE ATTACK (L inf)
        imgs_square = square.generate(x=imgs.cpu().numpy(), y=labels.cpu().numpy())
        ##ART returns numpy, need to convert imgs_square to tensor to pass it to the model
        imgs_square = torch.from_numpy(imgs_square).float().to(device)
        ##I normalize the images here, because Square needs not normalized images

        # NES ATTACK
        eps = 8/255
        imgs_nes = attack.generate(x=imgs.cpu().numpy(), y=labels.cpu().numpy())
        imgs_nes = torch.from_numpy(imgs_nes).float().to(device)


        # normalize
        imgs_square = normalize(imgs_square)
        imgs = normalize(imgs)
        imgs_nes = normalize(imgs_nes)

        # inferenza
        with torch.no_grad():
            logits_clean = model(imgs)
            logits_square = model(imgs_square)
            logits_nes = model(imgs_nes)

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            clean_metrics.update(labels, probs_clean)
            probs_square = torch.softmax(logits_square, dim=1)[:,1].detach().cpu().numpy()
            square_metrics.update(labels, probs_square)
            probs_nes = torch.softmax(logits_nes, dim=1)[:,1].detach().cpu().numpy()
            nes_metrics.update(labels, probs_nes)
    
    #Attack success 
    square_metrics.attack_success_rate(clean_metrics.all_probs)
    nes_metrics.attack_success_rate(clean_metrics.all_probs)
    
    clean_results = clean_metrics.compute()
    square_results = square_metrics.compute()
    nes_results = nes_metrics.compute()

    print("CLEAN RESULTS")
    clean_metrics.print(0)
    print("SQUARE ATTACK RESULTS (black-box)")
    square_metrics.print(0)
    print(f"Attack Success Rate:  {square_metrics.asr_list[0]}")

    print("NES ATTACK RESULTS (black-box)")
    nes_metrics.print(0)
    print(f"Attack Success Rate:  {nes_metrics.asr_list[0]}")


    return clean_metrics, square_metrics, nes_metrics

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=None)
    # I modify the last layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    model_name = "resnet50_clean_epoch_30_LR_0.0003_batchsize_32_WD_1e-05"

    # I load the trained clean model
    checkpoint_path = f"models_10/clean_resnet50/{model_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
        
    print("Initializing testing dataset....")
    test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
    # I get a smaller subset of 500 images
    test_small = balanced_subset(test_dataset, n_per_class=4)
    test_small_zoo = balanced_subset(test_dataset, n_per_class=25)
    
    print("Initializing test loader...")
    test_loader = DataLoader(test_small, batch_size=8, shuffle=True)
    test_loader_zoo = DataLoader(test_small_zoo, batch_size=BATCH_SIZE, shuffle=True)

    clean_metrics, square_metrics, nes_metrics = test_black(model, test_loader, model_name, device)
    #clean_metrics_zoo, zoo_metrics = test_zoo(model, test_loader_zoo, device)

    #saving metrics history
    history = {
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
        "square_epsilon": TEST_EPS_SQUARE,
        "nes_auc": nes_metrics.auc_list,
        "nes_f1": nes_metrics.f1_list,
        "nes_precision": nes_metrics.precision_list,
        "nes_recall": nes_metrics.recall_list,
        "nes_accuracy": nes_metrics.accuracy_list,
        "nes_asr": nes_metrics.asr_list
    }
    save_history_json(history,f"test_results_black/{model_name}/results_robust_iter_{SQUARE_ITER}_EPS_{TEST_EPS_SQUARE}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}.json")


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
    #plot_roc(clean_metrics.fpr, clean_metrics.tpr, clean_metrics.auc_list[0], "(test)", "clean_test")
    #plot_roc(square_metrics.fpr, square_metrics.tpr, square_metrics.auc_list[0], "(test)", f"square_test_{SQUARE_ITER}_iterations")


