import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision.models import resnet50, ResNet50_Weights
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

def train_robust_with_entropy(model, train_loader, val_loader, epsilon, start_epoch, num_epochs, optimizer, scheduler, criterion, device, train_losses, train_metrics_clean, train_metrics_adv, val_metrics, seed):
    train_loss = 0.0
    history = {}
    early_stopping = EarlyStopping(patience=100)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
    )


    for epoch in range(start_epoch, num_epochs):
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
            imgs_raw, y = batch
            imgs_raw, y = imgs_raw.to(device), y.to(device).long().squeeze()
            
            # Clean Forward pass
            optimizer.zero_grad()
            imgs_raw.requires_grad = True
            imgs = normalize(imgs_raw)
            logits_clean = model(imgs)
            loss_clean = criterion(logits_clean,y)

            #ADVERSARIAL TRAINING on FGSM radom start (robust to gradient masking)
            #I compute the gradient respect to the image
            #How much does the clean_loss change if I change the input image
            grad_imgs = torch.autograd.grad(loss_clean, imgs_raw, retain_graph=True, create_graph=False)[0]
            imgs_adv = imgs_raw + epsilon * grad_imgs.sign()
            imgs_adv = torch.clamp(imgs_adv, 0, 1)
            #imgs_adv = torch.clamp(imgs_adv, imgs - epsilon, imgs + epsilon)
            imgs_adv = normalize(imgs_adv.detach())
            
            #Adversarial forward pass
            logits_adv = model(imgs_adv)
            loss_adv = criterion(logits_adv, y)

            # compute entropy
            entropy_clean = entropy_penalty(logits_clean)
            entropy_adv = entropy_penalty(logits_adv)
            # adding entropy penalty to the loss
            loss = 0.5 * (loss_clean + loss_adv) - LAMBDA_ENTROPY * (entropy_clean + entropy_adv) / 2
            train_loss += loss.item() * imgs.size(0)
            epoch_loss = train_loss / len(train_loader.dataset)
            loss.backward()
            optimizer.step()
            scheduler.step()

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
        val_metrics_clean.reset_epoch()
        val_metrics_adv.reset_epoch()
        val_loss = 0.0

        pbar = tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
            if batch is None:
                continue
            imgs_raw, y = batch
            imgs_raw, y = imgs_raw.to(device), y.to(device).long().squeeze()
            # Adversarial validation
            imgs_raw.requires_grad = True
            logits_clean = model(normalize(imgs_raw))
            grad_imgs = torch.autograd.grad(loss_clean, imgs_raw, retain_graph=True, create_graph=False)[0]
            imgs_adv = imgs_raw +  current_eps * grad_imgs.sign()
            imgs_adv = torch.clamp(imgs_adv, 0, 1)

            with torch.no_grad():
                logits_clean = model(normalize(imgs_raw))
                loss_clean = criterion(logits_clean,y)
                logits_adv = model(normalize(imgs_adv))
                loss_adv = criterion(logits_adv,y)
                # compute entropy
                entropy_clean = entropy_penalty(logits_clean)
                entropy_adv = entropy_penalty(logits_adv)
                # adding entropy penalty to the loss
                loss = 0.5 * (loss_clean + loss_adv) - LAMBDA_ENTROPY * (entropy_clean + entropy_adv) / 2
                val_loss += loss.item() * imgs.size(0)
                epoch_val_loss = val_loss / len(val_loader.dataset)
    
                probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_clean.update(y, probs_clean)
                probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_adv.update(y, probs_adv)

        val_results_clean = val_metrics_clean.compute()
        val_results_adv = val_metrics_adv.compute()

        print(f"Epoch {epoch+1}:")
        print("TRAINING")
        print("Training loss:", epoch_loss)
        print("CLEAN RESULTS")
        train_metrics_clean.print(epoch)
        print("FGSM RESULTS")
        train_metrics_adv.print(epoch)
        print("ATTACK SUCCESS RATE")
        print(train_metrics_adv.asr_list[epoch])

        print("CLEAN VALIDATION")
        val_metrics_clean.print(epoch)
        print("ADV VALIDATION")
        val_metrics_adv.print(epoch)

        print("softmax mean clean:", torch.softmax(logits_clean, dim=1).mean(0).detach().cpu().tolist())
        print("Entropy clean:", entropy_clean.item())
        print("softmax mean adv:", torch.softmax(logits_adv, dim=1).mean(0).detach().cpu().tolist())
        print("Entropy adv:", entropy_adv.item())

        # early stopping
        #print(f"Epoch {epoch+1} - val_loss: {epoch_val_loss:.4f}")
        #if early_stopping(epoch_val_loss, model, optimizer, epoch, seed, "square", train_metrics_clean, train_metrics_adv, val_metrics, train_losses):
        #    break

    model_path = f'{MODELS_DIR}/no_eps_scheduler/resnet50_square_epoch_{epoch}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_EPS_{epsilon}_seed_{seed}_None_sched_3.pt'
    ##SALVA I PESI DEL MODELLO
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        'train_losses': train_losses,
        "train_auc_clean": train_metrics_clean.auc_list,
        "train_auc_adv": train_metrics_adv.auc_list,
        "val_auc_clean": val_metrics_clean.auc_list,
        "val_auc_adv": val_metrics_adv.auc_list,
        "train_tpr_clean": train_metrics_clean.tpr,
        "train_tpr_adv": train_metrics_adv.tpr,
        "val_tpr_clean": val_metrics_clean.tpr,
        "val_tpr_adv": val_metrics_adv.tpr,
        "train_fpr_clean": train_metrics_clean.fpr,
        "train_fpr_adv": train_metrics_adv.fpr,
        "val_fpr_clean": val_metrics_clean.fpr,
        "val_fpr_adv": val_metrics_adv.fpr
    },model_path)
    # save history
    history = {
        "epoch": epoch+1,
        "train_losses": train_losses,
        "train_auc_clean": train_metrics_clean.auc_list,
        "train_auc_adv": train_metrics_adv.auc_list,
        "val_auc_clean": val_metrics_clean.auc_list,
        "train_f1_clean": train_metrics_clean.f1_list,
        "train_f1_adv": train_metrics_adv.f1_list,
        "val_f1_clean": val_metrics_clean.f1_list,
        "train_precision_clean": train_metrics_clean.precision_list,
        "train_precision_adv": train_metrics_adv.precision_list,
        "val_precision_clean": val_metrics_clean.precision_list,
        "train_recall_clean": train_metrics_clean.recall_list,
        "train_recall_adv": train_metrics_adv.recall_list,
        "val_recall_clean": val_metrics_clean.recall_list,
        "train_accuracy_clean": train_metrics_clean.accuracy_list,
        "train_accuracy_adv": train_metrics_adv.accuracy_list,
        "val_accuracy_clean": val_metrics_clean.accuracy_list,
        "val_auc_adv": val_metrics_adv.auc_list,
        "val_f1_adv": val_metrics_adv.f1_list,
        "val_precision_adv": val_metrics_adv.precision_list,
        "val_recall_adv": val_metrics_adv.recall_list,
        "val_accuracy_adv": val_metrics_adv.accuracy_list,
        "train_asr": train_metrics_adv.asr_list,
        "train_epsilon": current_eps
    }

    save_history_json(history,f"history/history_square/no_eps_scheduler/history_square_epoch_{epoch}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_EPS_{epsilon}_seed_{seed}_None_sched_3.json")

    return train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv train_losses

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    seed=42
    set_seed(seed)
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, 2)
    )
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
    #print("Counting labels...")
    #train_counts = count_labels(train_dataset)
    #print("Counting done")
    #num_real_train = train_counts[0]
    #num_fake_train = train_counts[1]
    #print(num_fake_train)
    #print(num_real_train)
    #pos_weight = num_real_train/(num_fake_train + num_real_train)
    #neg_weight = num_fake_train/(num_fake_train + num_real_train)
    #class_weights = torch.tensor([pos_weight, neg_weight]).to(device)
    # Cross entropy loss with label smoothing
    #criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=1e-7, 
        max_lr=1e-4, 
        step_size_up=500, 
        mode='triangular2'
        )
    train_metrics_clean = Metrics()
    train_metrics_adv = Metrics()
    val_metrics = Metrics()
    start_epoch = 0
    train_losses = []

    # Starting robust training from the pre-trained clean model
    #checkpoint_path = f"{MODELS_DIR}/resnet50_square_epoch_4_LR_0.0001_batchsize_64_WD_0.01_EPS_0.01568627450980392_fine_tuned.pt"
    checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #start_epoch = checkpoint['epoch'] # riparte dall'epoch successivo
    #print(f"Riprendo dal epoch {start_epoch}")
    #train_losses = checkpoint["train_losses"]
#
    #train_metrics_clean.auc_list = checkpoint["train_auc_clean"]
    #train_metrics_adv.auc_list = checkpoint["train_auc_adv"]
    #val_metrics.auc_list = checkpoint["val_auc"]
    #
    #train_metrics_clean.tpr = checkpoint["train_tpr_clean"]
    #train_metrics_adv.tpr = checkpoint["train_tpr_adv"]
    #val_metrics.tpr = checkpoint["val_tpr"]
    #
    #train_metrics_clean.fpr = checkpoint["train_fpr_clean"]
    #train_metrics_adv.fpr = checkpoint["train_fpr_adv"]
    #val_metrics.fpr = checkpoint["val_fpr"]
#
    #history_path = "history/history_square/history_square_epoch_4_LR_0.0001_batchsize_64_WD_0.01_EPS_0.01568627450980392.json"
    #with open(history_path, "r") as f:
    #    history = json.load(f)
    #
    #train_metrics_clean.f1_list = history["train_f1_clean"]
    #train_metrics_adv.f1_list = history["train_f1_adv"]
    #val_metrics.f1_list = history["val_f1"]
    #
    #train_metrics_clean.precision_list = history["train_precision_clean"]
    #train_metrics_adv.precision_list = history["train_precision_adv"]
    #val_metrics.precision_list = history["val_precision"]
    #
    #train_metrics_clean.recall_list = history["train_recall_clean"]
    #train_metrics_adv.recall_list = history["train_recall_adv"]
    #val_metrics.recall_list = history["val_recall"]
    #
    #train_metrics_clean.accuracy_list = history["train_accuracy_clean"]
    #train_metrics_adv.accuracy_list = history["train_accuracy_adv"]
    #val_metrics.accuracy_list = history["val_accuracy"]
#
    #train_metrics_adv.asr_list = history["train_asr"]
#
    epsilons = [2/255, 8/255]
    for eps in epsilons:
        train_metrics_clean = Metrics()
        train_metrics_adv = Metrics()
        val_metrics = Metrics()
        start_epoch = 0
        train_losses = []

        print(f"Starting training with epsilon {eps}")
        train_metrics_clean, train_metrics_adv, val_metrics, train_losses = train_robust_with_entropy(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader,
            epsilon=eps,
            start_epoch=start_epoch, 
            num_epochs=12, 
            optimizer=optimizer, 
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            train_losses=train_losses,
            train_metrics_clean=train_metrics_clean,
            train_metrics_adv=train_metrics_adv,
            val_metrics=val_metrics,
            seed=seed) 

        plot_roc(val_metrics.fpr, val_metrics.tpr, val_metrics.auc_list[11], 12, 
                     f"plots/square/ROC_plot_numepochs_11_LR_{LR}_batchsize{BATCH_SIZE}_WD_{WD}_EPS_{eps}_None_sched.png")
        
        #folder_name = "square"
        #file_name = ""
        ##plot loss
        #plot_metric(train_metrics_clean.accuracy_list, val_metrics.accuracy_list, 12, "Accuracy", 
        #    f"plots/{folder_name}/Accuracy_plot_{file_name}.png")
        ##plot f1 score
        #plot_metric(train_metrics_clean.f1_list,  val_metrics.f1_list, 12, "F1_score", 
        #            f"plots/{folder_name}/F1_plot_{file_name}.png")
        ##plot precision
        #plot_metric(train_metrics_clean.precision_list,  val_metrics.precision_list, 12, "Precision",
        #            f"plots/{folder_name}/Precision_plot_{file_name}.png")
        ##plot recall
        #plot_metric(train_metrics_clean.recall_list,  val_metrics.recall_list, 12, "Recall", 
        #    f"plots/{folder_name}/Recall_plot_{file_name}.png")
#
        #plot_loss(train_metrics_clean.train_losses, f"plots/{folder_name}/Loss_plot_{file_name}.png")
        ##plot AUC
        
    