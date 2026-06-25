import os
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import foolbox as fb

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Class for computing metrics
class Metrics:
    def __init__(self):
        self.y_true = 0
        self.y_pred = 0
        self.y_prob = 0
        self.fpr = 0
        self.tpr = 0
        self.total_l2 = 0
        self.total_linf = 0
        self.avg_l2 = 0
        self.avg_linf = 0
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
    
        # successful attacks - the attack changes the prediction (in general)
        #successful_attacks = correct_clean & (pred_adv != y_true)
        successful_attacks = pred_adv != pred_clean
    
        if correct_clean.sum() == 0:
            return 0.0  # no division by 0
        
        #asr = successful_attacks.sum() / correct_clean.sum()
        asr = successful_attacks.sum() / len(pred_adv)
        self.asr_list.append(asr)

    
    def print(self, epoch):
        print(f"Accuracy:  {self.accuracy_list[epoch]:.4f}")
        print(f"F1 score:   {self.f1_list[epoch]:.4f}")
        print(f"Precision:   {self.precision_list[epoch]:.4f}")
        print(f"Recall:   {self.recall_list[epoch]:.4f}")
        print(f"AUC score:   {self.auc_list[epoch]:.4f}")

def entropy_penalty(logits, eps=1e-8):
    #logits: (B, C)
    #returns: scalar entropy (mean over batch)
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=1)  # (B,)
    return entropy.mean()


def train_robust(model, train_loader, val_loader, epsilon, start_epoch, num_epochs, optimizer, scheduler, criterion, device, train_losses, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, seed, sched_type, entropy_flag=False, has_eps_sched=False):
    train_loss = 0.0
    history = {}
    val_losses = []
    early_stopping = EarlyStopping(patience=100)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])


    denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    eps_scheduler = EpsilonScheduler(eps_start=0/255, eps_end=8/255, num_epochs_rampup=10, type=sched_type)
    print(f"Training with {eps_scheduler.type} epsilon scheduler")

    for epoch in range(start_epoch, num_epochs):
        #TRAINING
        model.train()
        train_loss = 0.0
        if has_eps_sched == False:
            current_eps = epsilon
        else:
            current_eps = eps_scheduler.get_epsilon(epoch)

        # I block update statistics of BatchNorm
        freeze_bn(model)
        train_metrics_clean.reset_epoch()
        train_metrics_adv.reset_epoch()
        val_metrics_clean.reset_epoch()
        val_metrics_adv.reset_epoch()
        print(f"Epoch {epoch+1} | eps: {current_eps:.4f}")

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            if batch is None:
                continue
            imgs_raw, y = batch
            imgs_raw, y = imgs_raw.to(device), y.to(device).long().squeeze()
            #print(f"imgs_raw min/max: {imgs_raw.min():.3f}/{imgs_raw.max():.3f}")
            #print(f"imgs min/max: {imgs.min():.3f}/{imgs.max():.3f}")

            # Clean Forward pass
            optimizer.zero_grad()
            imgs_raw.requires_grad = True
            imgs = normalize(imgs_raw)
            logits_clean = model(imgs)
            loss_clean = criterion(logits_clean,y)
             
            #I compute the gradient respect to the image
            #How much does the clean_loss change if I change the input image
            grad_imgs = torch.autograd.grad(loss_clean, imgs_raw, retain_graph=True, create_graph=False)[0]
            imgs_adv = imgs_raw +  current_eps * grad_imgs.sign()
            imgs_adv = torch.clamp(imgs_adv, 0, 1)
            #imgs_adv = torch.clamp(imgs_adv, imgs - current_eps, imgs + current_eps)
            imgs_adv = normalize(imgs_adv.detach())
            
            # I compute again the clean loss with fresh graph
            #imgs_detached = imgs.detach()
            #imgs_detached.requires_grad = False

            #print(f"imgs_adv min/max: {imgs_adv.min():.3f}/{imgs_adv.max():.3f}")

            #Adversarial forward pass
            logits_adv = model(imgs_adv) #detach
            loss_adv = criterion(logits_adv, y)

            if entropy_flag == True:
                # compute entropy
                entropy_clean = entropy_penalty(logits_clean)
                entropy_adv = entropy_penalty(logits_adv)
                # adding entropy penalty to the loss
                loss = 0.5 * (loss_clean + loss_adv) - LAMBDA_ENTROPY * (entropy_clean + entropy_adv) / 2
            else:
                loss = 0.5 * loss_clean + 0.5 * loss_adv

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
            # Costruzione attacco - richiede il grafo
            with torch.enable_grad():
                imgs_raw = imgs_raw.detach().requires_grad_(True)
                imgs_norm = normalize(imgs_raw)
                logits_clean = model(imgs_norm)
                loss_tmp = criterion(logits_clean, y)
                grad_imgs = torch.autograd.grad(loss_tmp, imgs_raw)[0]

            imgs_adv = imgs_raw + current_eps * grad_imgs.sign()
            imgs_adv = torch.clamp(imgs_adv, 0, 1)

            with torch.no_grad():
                logits_clean = model(normalize(imgs_raw.detach()))
                loss_clean = criterion(logits_clean,y)
                logits_adv = model(normalize(imgs_adv.detach()))
                loss_adv = criterion(logits_adv,y)
                if entropy_flag == True:
                    # compute entropy
                    entropy_clean = entropy_penalty(logits_clean)
                    entropy_adv = entropy_penalty(logits_adv)
                    # adding entropy penalty to the loss
                    loss = 0.5 * (loss_clean + loss_adv) - LAMBDA_ENTROPY * (entropy_clean + entropy_adv) / 2
                else:
                    loss = 0.5 * loss_clean + 0.5 * loss_adv

                val_loss += loss.item() * imgs.size(0)
                epoch_val_loss = val_loss / len(val_loader.dataset)
    
                probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_clean.update(y, probs_clean)
                probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_adv.update(y, probs_adv)

        val_results_clean = val_metrics_clean.compute()
        val_results_adv = val_metrics_adv.compute()
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}:")
        print("TRAINING")
        print("Training loss:", epoch_loss)
        print("Validation loss:", epoch_val_loss)
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

        
    attack_name = ""
    if entropy_flag == True:
        attack_name = "square"
    else:
        attack_name = "fgsm"

    eps_sched_folder = ""
    save_path = ""
    history_path = ""
    
    if has_eps_sched == True:
        eps_sched_folder = "with_eps_scheduler"
        save_path =  f'{MODELS_DIR}/{eps_sched_folder}/resnet50_{attack_name}_epoch_{epoch}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_seed_{seed}_{eps_scheduler.type}_sched_3.pt'
        history_path = f"history/history_{attack_name}/{eps_sched_folder}/history_{attack_name}_epoch_{epoch}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_seed_{seed}_{eps_scheduler.type}_sched_3.json"
    else:
        eps_sched_folder = "no_eps_scheduler"
        save_path =  f'{MODELS_DIR}/{eps_sched_folder}/resnet50_{attack_name}_epoch_{epoch}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_EPS_{current_eps}_seed_{seed}_{eps_scheduler.type}_sched_3_prova.pt'
        history_path = f"history/history_{attack_name}/{eps_sched_folder}/history_{attack_name}_epoch_{epoch}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_EPS_{current_eps}_seed_{seed}_{eps_scheduler.type}_sched_3_prova.json"
        

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
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
    },save_path)
    # save history
    history = {
        "epoch": epoch+1,
        "train_losses": train_losses,
        'val_losses': val_losses,
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

    save_history_json(history, history_path)
    

    return train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    set_seed(42)
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)
    
    #transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(
    #        mean=[0.485, 0.456, 0.406],
    #        std=[0.229, 0.224, 0.225])
    #   ])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
        
    print("Initializing training dataset....")
    train_dataset = FFDataset(root_dir=ROOT_DIR, split="train", transform=transform)
    # I get a small subset for debugging
    #train_small = balanced_subset(train_dataset, n_per_class=2)
    
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
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
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
    #history_path = "history_fgsm/history_fgsm_epoch_10_LR_0.0003_batchsize_32_WD_1e-05.json"
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

    scheduler_type = ['cosine', 'linear']
    epsilons = [8/255]

    # TRAIN WITH EPSILON SCHEDULER
    #for type in scheduler_type:
    #    train_metrics_clean = Metrics()
    #    train_metrics_adv = Metrics()
    #    val_metrics_clean = Metrics()
    #    val_metrics_adv = Metrics()
    #    start_epoch = 0
    #    train_losses = []
    #    
    #    # FGSM-AT + entropy penalty
    #    train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses = train_robust(
    #        model=model, 
    #        train_loader=train_loader, 
    #        val_loader=val_loader,
    #        epsilon=0,
    #        start_epoch=start_epoch, 
    #        num_epochs=12, 
    #        optimizer=optimizer,
    #        scheduler=scheduler, 
    #        criterion=criterion,
    #        device=device,
    #        train_losses=train_losses,
    #        train_metrics_clean=train_metrics_clean,
    #        train_metrics_adv=train_metrics_adv,
    #        val_metrics_clean=val_metrics_clean,
    #        val_metrics_adv=val_metrics_adv,
    #        seed=seed,
    #        sched_type=type,
    #        entropy_flag=True,
    #        has_eps_sched=True)

    #    train_metrics_clean = Metrics()
    #    train_metrics_adv = Metrics()
    #    val_metrics_clean = Metrics()
    #    val_metrics_adv = Metrics()
    #    start_epoch = 0
    #    train_losses = []

    #    # FGSM-AT
    #    train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses = train_robust(
    #        model=model, 
    #        train_loader=train_loader, 
    #        val_loader=val_loader,
    #        epsilon=0,
    #        start_epoch=start_epoch, 
    #        num_epochs=12, 
    #        optimizer=optimizer,
    #        scheduler=scheduler, 
    #        criterion=criterion,
    #        device=device,
    #        train_losses=train_losses,
    #        train_metrics_clean=train_metrics_clean,
    #        train_metrics_adv=train_metrics_adv,
    #        val_metrics_clean=val_metrics_clean,
    #        val_metrics_adv=val_metrics_adv,
    #        seed=seed,
    #        sched_type=type,
    #        entropy_flag=False,
    #        has_eps_sched=True)
    # TRAIN WITH FIXED EPSILON
    for eps in epsilons:
        #train_metrics_clean = Metrics()
        #train_metrics_adv = Metrics()
        #val_metrics_clean = Metrics()
        #val_metrics_adv = Metrics()
        #start_epoch = 0
        #train_losses = []
        #
        #print(f"Starting FGSM-AT + entropy penalty with eps={eps}")
        ## FGSM-AT + entropy penalty
        #train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses = train_robust(
        #    model=model, 
        #    train_loader=train_loader, 
        #    val_loader=val_loader,
        #    epsilon=eps,
        #    start_epoch=start_epoch, 
        #    num_epochs=12, 
        #    optimizer=optimizer,
        #    scheduler=scheduler, 
        #    criterion=criterion,
        #    device=device,
        #    train_losses=train_losses,
        #    train_metrics_clean=train_metrics_clean,
        #    train_metrics_adv=train_metrics_adv,
        #    val_metrics_clean=val_metrics_clean,
        #    val_metrics_adv=val_metrics_adv,
        #    seed=seed,
        #    sched_type='None',
        #    entropy_flag=True,
        #    has_eps_sched=False)
        train_metrics_clean = Metrics()
        train_metrics_adv = Metrics()
        val_metrics_clean = Metrics()
        val_metrics_adv = Metrics()
        start_epoch = 0
        train_losses = []
        # FGSM-AT
        print(f"Starting FGSM-AT with eps={eps}")
        train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses = train_robust(
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
            val_metrics_clean=val_metrics_clean,
            val_metrics_adv=val_metrics_adv,
            seed=seed,
            sched_type='None',
            entropy_flag=False,
            has_eps_sched=False)


        
