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

def train_robust(model, train_loader, val_loader, start_epoch, num_epochs, optimizer, criterion, device, train_losses, train_metrics, val_metrics):
    train_loss = 0.0
    history = {}
    fmodel = fb.PyTorchModel(model, bounds=(0,1), device=device)

    for epoch in range(start_epoch, NUM_EPOCHS):
        #TRAINING
        model.train()
        train_metrics.reset_epoch()
        val_metrics.reset_epoch()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            if batch is None:
                continue
            imgs, y = batch
            imgs, y = imgs.to(device), y.to(device).float().unsqueeze(1)

            #In order to generate FGSM we need the gradients
            imgs.requires_grad = True
            
            #ADVERSARIAL TRAINING on FGSM radom start (robust to gradient masking)
            x = imgs + torch.empty_like(imgs).uniform_(-EPS, EPS)
            x = torch.clamp(x, 0, 1)

            x.requires_grad = True
            logits = model(x)
            loss = criterion(logits, y)

            grad = torch.autograd.grad(loss, x)[0]
            adv_imgs = x + EPS * grad.sign()
            adv_imgs = torch.clamp(adv_imgs, 0, 1)

            optimizer.zero_grad()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            train_metrics.update(y, probs)

        train_losses.append(train_loss/len(train_loader.dataset))
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
                imgs, y = imgs.to(device), y.to(device).float().unsqueeze(1)
                logits = model(imgs)
    
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                val_metrics.update(y, probs)

        val_results = val_metrics.compute()

        print(f"Epoch {epoch+1}:")
        print("TRAINING")
        print("Training loss:", train_loss)
        train_metrics.print(epoch)

        print("VALIDATION")
        val_metrics.print(epoch)


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
        }, f'models/fgsm_resnet50/resnet50_clean_epoch_{epoch+1}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}.pt')
        print("Model saved in models/fgsm_resnet50")

       # saving metrics history
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
        save_history_json(history,f"history_fgsm/history_fgsm_epoch_{epoch+1}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}.json")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=None)
    # I modify the last layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)
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
    pos_weight = torch.tensor([num_real_train/num_fake_train]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    train_metrics = Metrics()
    val_metrics = Metrics()
    start_epoch = 0
    train_losses = []

    # Starting robust training from the pre-trained clean model
    checkpoint_path = "models/clean_resnet50/resnet50_clean_epoch_2_LR_0.0001_batchsize_16_WD_0.0001.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_metrics, val_metrics, train_losses = train_robust(
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
    