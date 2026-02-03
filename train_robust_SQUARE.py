from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
os.environ.pop("SSLKEYLOGFILE", None)
import foolbox as fb

def entropy_penalty(logits, eps=1e-8):
    #logits: (B, C)
    #returns: scalar entropy (mean over batch)
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=1)  # (B,)
    return entropy.mean()

def train_robust(model, train_loader, val_loader, start_epoch, num_epochs, optimizer, criterion, device, train_losses, train_metrics_clean, train_metrics_adv, val_metrics):
    train_loss = 0.0
    history = {}

    # I define the image bounds for the fmodel in order to properly attack in that space
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    lower = (0 - mean) / std
    upper = (1 - mean) / std
    #fmodel = fb.PyTorchModel(model, bounds=(lower.min().item(), upper.max().item()), device=device)

    for epoch in range(start_epoch, NUM_EPOCHS):
        #TRAINING
        model.train()
        train_loss = 0.0
        # I block update statistics of BatchNorm
        freeze_bn(model)
        train_metrics_clean.reset_epoch()
        train_metrics_adv.reset_epoch()
        val_metrics.reset_epoch()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            if batch is None:
                continue
            imgs, y = batch
            imgs, y = imgs.to(device), y.to(device).long().squeeze()
            
            # Clean Forward pass
            imgs.requires_grad = True
            logits_clean = model(imgs)
            loss_clean = criterion(logits_clean,y)

            #ADVERSARIAL TRAINING on FGSM radom start (robust to gradient masking)
            #I compute the gradient respect to the image
            #How much does the clean_loss change if I change the input image
            grad_imgs = torch.autograd.grad(loss_clean, imgs, retain_graph=True, create_graph=False)[0]
            imgs_adv = imgs + EPS * grad_imgs.sign()
            imgs_adv = torch.clamp(imgs_adv, 0, 1)
            
            #Adversarial forward pass
            logits_adv = model(imgs_adv)
            loss_adv = criterion(logits_adv, y)

            optimizer.zero_grad()

            # compute entropy
            entropy_clean = entropy_penalty(logits_clean)
            entropy_adv = entropy_penalty(logits_adv)
            # adding entropy penalty to the loss
            loss = 0.5 * (loss_clean + loss_adv) - LAMBDA_ENTROPY * (entropy_clean + entropy_adv) / 2
            train_loss += loss.item() * imgs.size(0)
            epoch_loss = train_loss / len(train_loader.dataset)
            loss.backward()
            optimizer.step()

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            train_metrics_clean.update(y, probs_clean)
            probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
            train_metrics_adv.update(y, probs_adv)

        train_losses.append(epoch_loss)
        train_results_clean = train_metrics_clean.compute()
        train_results_adv = train_metrics_adv.compute()
        train_metrics_adv.attack_success_rate(train_metrics_clean.all_probs)

        #VALIDATION
        model.eval()
        val_metrics.reset_epoch()

        pbar = tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}", unit="batch")
        with torch.no_grad():
            for batch in pbar:
                if batch is None:
                    continue
                imgs, y = batch
                imgs, y = imgs.to(device), y.to(device).long().squeeze()
                logits = model(imgs)
    
                probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
                val_metrics.update(y, probs)

        val_results = val_metrics.compute()

        print(f"Epoch {epoch+1}:")
        print("TRAINING")
        print("Training loss:", epoch_loss)
        print("CLEAN RESULTS")
        train_metrics_clean.print(epoch)
        print("FGSM RESULTS")
        train_metrics_adv.print(epoch)
        print("ATTACK SUCCESS RATE")
        print(train_metrics_adv.asr_list[epoch])

        print("VALIDATION")
        val_metrics.print(epoch)

        print("softmax mean clean:", torch.softmax(logits_clean, dim=1).mean(0).detach().cpu().tolist())
        print("Entropy clean:", entropy_clean.item())
        print("softmax mean adv:", torch.softmax(logits_adv, dim=1).mean(0).detach().cpu().tolist())
        print("Entropy adv:", entropy_adv.item())


        #SALVA I PESI DEL MODELLO
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            'loss': loss,
            'train_losses': train_losses,
            "train_auc_clean": train_metrics_clean.auc_list,
            "train_auc_adv": train_metrics_adv.auc_list,
            "val_auc": val_metrics.auc_list,
            "train_tpr_clean": train_metrics_clean.tpr,
            "train_tpr_adv": train_metrics_adv.tpr,
            "val_tpr": val_metrics.tpr,
            "train_fpr_clean": train_metrics_clean.fpr,
            "train_fpr_adv": train_metrics_adv.fpr,
            "val_fpr": val_metrics.fpr
        }, f'models/square_resnet50/resnet50_square_epoch_{epoch+1}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}.pt')
        print("Model saved in models/square_resnet50")

       # saving metrics history
        history = {
            "train_losses": train_losses,
            "train_auc_clean": train_metrics_clean.auc_list,
            "train_auc_adv": train_metrics_adv.auc_list,
            "val_auc": val_metrics.auc_list,
            "train_f1_clean": train_metrics_clean.f1_list,
            "train_f1_adv": train_metrics_adv.f1_list,
            "val_f1": val_metrics.f1_list,
            "train_precision_clean": train_metrics_clean.precision_list,
             "train_precision_adv": train_metrics_adv.precision_list,
            "val_precision": val_metrics.precision_list,
            "train_recall_clean": train_metrics_clean.recall_list,
            "train_recall_adv": train_metrics_adv.recall_list,
            "val_recall": val_metrics.recall_list,
            "train_accuracy_clean": train_metrics_clean.accuracy_list,
            "train_accuracy_adv": train_metrics_adv.accuracy_list,
            "val_accuracy": val_metrics.accuracy_list,
            "train_asr": train_metrics_adv.asr_list
        }

        save_history_json(history,f"history_square/history_square_epoch_{epoch+1}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}.json")

    return train_metrics_clean, train_metrics_adv, val_metrics, train_losses

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=None)
    # I modify the last layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
       ])
        
    print("Initializing training dataset....")
    train_dataset = FFDataset(root_dir=ROOT_DIR, split="train", transform=transform)
    # I get a small subset for debugging
    #train_small = balanced_subset(train_dataset, n_per_class=36)
    
    #print(train_dataset.getitem(0))
    print("Initializing validation dataset....")
    val_dataset = FFDataset(root_dir=ROOT_DIR, split="val", transform=transform)
    
    print("Initializing train loader...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Initializing val loader....")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Unbalanced dataset since 4000 fake videos and 1000 real
    # so balance fake vs real during training
    print("Counting labels...")
    train_counts = count_labels(train_dataset)
    print("Counting done")
    num_real_train = train_counts[0]
    num_fake_train = train_counts[1]
    print(num_fake_train)
    print(num_real_train)
    pos_weight = num_real_train/(num_fake_train + num_real_train)
    neg_weight = num_fake_train/(num_fake_train + num_real_train)
    class_weights = torch.tensor([pos_weight, neg_weight]).to(device)
    # Cross entropy loss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    train_metrics_clean = Metrics()
    train_metrics_adv = Metrics()
    val_metrics = Metrics()
    start_epoch = 0
    train_losses = []

    # Starting robust training from the pre-trained clean model
    checkpoint_path = "models/square_resnet50/resnet50_square_epoch_9_LR_0.0003_batchsize_32_WD_1e-05.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] # riparte dall'epoch successivo
    print(f"Riprendo dal epoch {start_epoch}")
    train_losses = checkpoint["train_losses"]

    train_metrics_clean.auc_list = checkpoint["train_auc_clean"]
    train_metrics_adv.auc_list = checkpoint["train_auc_adv"]
    val_metrics.auc_list = checkpoint["val_auc"]
    
    train_metrics_clean.tpr = checkpoint["train_tpr_clean"]
    train_metrics_adv.tpr = checkpoint["train_tpr_adv"]
    val_metrics.tpr = checkpoint["val_tpr"]
    
    train_metrics_clean.fpr = checkpoint["train_fpr_clean"]
    train_metrics_adv.fpr = checkpoint["train_fpr_adv"]
    val_metrics.fpr = checkpoint["val_fpr"]

    history_path = "history_square/history_square_epoch_9_LR_0.0003_batchsize_32_WD_1e-05.json"
    with open(history_path, "r") as f:
        history = json.load(f)
    
    train_metrics_clean.f1_list = history["train_f1_clean"]
    train_metrics_adv.f1_list = history["train_f1_adv"]
    val_metrics.f1_list = history["val_f1"]
    
    train_metrics_clean.precision_list = history["train_precision_clean"]
    train_metrics_adv.precision_list = history["train_precision_adv"]
    val_metrics.precision_list = history["val_precision"]
    
    train_metrics_clean.recall_list = history["train_recall_clean"]
    train_metrics_adv.recall_list = history["train_recall_adv"]
    val_metrics.recall_list = history["val_recall"]
    
    train_metrics_clean.accuracy_list = history["train_accuracy_clean"]
    train_metrics_adv.accuracy_list = history["train_accuracy_adv"]
    val_metrics.accuracy_list = history["val_accuracy"]

    train_metrics_adv.asr_list = history["train_asr"]

    train_metrics_clean, train_metrics_adv, val_metrics, train_losses = train_robust(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        start_epoch=start_epoch, 
        num_epochs=NUM_EPOCHS, 
        optimizer=optimizer, 
        criterion=criterion,
        device=device,
        train_losses=train_losses,
        train_metrics_clean=train_metrics_clean,
        train_metrics_adv=train_metrics_adv,
        val_metrics=val_metrics)
    

    #plot loss
    plot_loss(train_metrics_clean.train_losses)
    #plot accuracy
    plot_metric(train_metrics_clean.accuracy_list, val_metrics.accuracy_list, NUM_EPOCHS, "Accuracy", 
                f"metrics_images_square/Train_accuracy_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")
    #plot f1 score
    plot_metric(train_metrics_adv.f1_list,  val_metrics.f1_list, NUM_EPOCHS, "F1_score", 
                f"metrics_images_square/Train_F1_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")
    #plot precision
    plot_metric(train_metrics_adv.precision_list,  val_metrics.precision_list, NUM_EPOCHS, "Precision",
                f"metrics_images_square/Train_precision_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")
    #plot recall
    plot_metric(train_metrics_clean.recall_list,  val_metrics.recall_list, NUM_EPOCHS, "Recall", 
                f"metrics_images_square/Train_recall_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")
    #plot AUC
    plot_roc(val_metrics.fpr, val_metrics.tpr, val_metrics.auc_list[NUM_EPOCHS-1], NUM_EPOCHS, 
             f"metrics_images_square/Train_ROC_plot_numepochs_{NUM_EPOCHS}_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}.png")
    