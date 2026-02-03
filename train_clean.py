from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


def train_clean(model, train_loader, val_loader, start_epoch, num_epochs, optimizer, criterion, device, train_losses, train_metrics, val_metrics):
    history = {}

    for epoch in range(start_epoch, num_epochs):
        #TRAINING
        model.train()
        train_metrics.reset_epoch()
        val_metrics.reset_epoch()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            if batch is None:
                continue
            imgs, y = batch
            #for CE loss labels must be (B,) so 1D with dtype torch.long
            imgs, y = imgs.to(device), y.to(device).long().squeeze()
            #print(y.shape, y.dtype, y.min(), y.max())
            logits = model(imgs)
            #print(logits.shape)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            epoch_loss = train_loss / len(train_loader.dataset)
            probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
            train_metrics.update(y, probs)

        train_losses.append(epoch_loss)
        train_results = train_metrics.compute()

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
        train_metrics.print(epoch)

        print("VALIDATION")
        val_metrics.print(epoch)

        print(probs.min(),probs.mean(),probs.max())
        print("softmax mean:", torch.softmax(logits, dim=1).mean(0))


        #SALVA I PESI DEL MODELLO
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            'loss': train_loss,
            'train_losses': train_losses,
            "train_auc": train_metrics.auc_list,
            "val_auc": val_metrics.auc_list,
            "train tpr": train_metrics.tpr,
            "val tpr": val_metrics.tpr,
            "train fpr": train_metrics.fpr,
            "val fpr": val_metrics.fpr
        }, f'models/clean_resnet50/resnet50_clean_epoch_{epoch+1}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_2.pt')
        print("Model saved in models/clean_resnet50")
        #saving metrics history
        history = {
            "train_losses": train_losses,
            "train_auc": train_metrics.auc_list,
            "val_auc": val_metrics.auc_list,
            "train_f1": train_metrics.f1_list,
            "val_f1": val_metrics.f1_list,
            "train_precision": train_metrics.precision_list,
            "val_precision": val_metrics.precision_list,
            "train_recall": train_metrics.recall_list,
            "val_recall": val_metrics.recall_list,
            "train_accuracy": train_metrics.accuracy_list,
            "val_accuracy": val_metrics.accuracy_list
        }
        save_history_json(history,f"history_clean/history_clean_epoch_{epoch+1}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_2.json")

    return train_metrics, val_metrics, train_losses

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
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    train_metrics = Metrics()
    val_metrics = Metrics()
    start_epoch = 0
    train_losses = []

    # ricarico il checkpoint
    checkpoint_path = "models/clean_resnet50/resnet50_clean_epoch_16_LR_0.0003_batchsize_32_WD_1e-05_2.pt"
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

    history_path = "history_clean/history_clean_epoch_16_LR_0.0003_batchsize_32_WD_1e-05_2.json"
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
    
    
    train_metrics, val_metrics, train_losses = train_clean(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        start_epoch=start_epoch, 
        num_epochs=NUM_EPOCHS, 
        optimizer=optimizer, 
        criterion=criterion,
        device=device,
        train_losses=train_losses,
        train_metrics=train_metrics,
        val_metrics=val_metrics)
    

    #plot loss
    plot_loss(train_metrics.train_losses)
    #plot accuracy
    plot_metric(train_metrics.accuracy_list, val_metrics.accuracy_list, "Accuracy")
    #plot f1 score
    plot_metric(train_metrics.f1_list,  val_metrics.f1_list, "F1_score")
    #plot precision
    plot_metric(train_metrics.precision_list,  val_metrics.precision_list, "Precision")
    #plot recall
    plot_metric(train_metrics.recall_list,  val_metrics.recall_list, "Recall")
    #plot AUC
    plot_roc(val_metrics.fpr, val_metrics.tpr, val_metrics.auc_list[NUM_EPOCHS-1], NUM_EPOCHS)
    