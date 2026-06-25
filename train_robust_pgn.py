import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "2"
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import *
import foolbox as fb
from test_attacks import test_attack

# FGSM-AT with curriculum epsilon scheduler and gradient norm penalty
def train_robust_with_curriculum_pgn(model, train_loader, val_loader, lambda_entropy, lambda_grad, start_epoch, num_epochs, optimizer, scheduler, eps_scheduler, criterion, device, train_losses, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, seed, base_lr, max_lr, alpha_adv, entropy_flag=False, adaptive=False, save_model=True):
    train_loss = 0.0
    history = {}
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    val_acc_adv = 0
    clean_acc   = 0
    
    print(f"Training with {eps_scheduler.type} epsilon scheduler")

    for epoch in range(start_epoch, num_epochs):
        #TRAINING
        model.train()
        train_loss = 0.0

        eps_cpunter = 0

        if eps_scheduler.should_stop():
                break
        else:
            current_eps = eps_scheduler.get_epsilon(val_acc_adv, clean_acc, epoch)

        # Traying uniform epsilon scheduler
        #current_eps = torch.FloatTensor(1).uniform_(2/255, 8/255).item()


        # I block update statistics of BatchNorm
        freeze_bn(model)
        train_metrics_clean.reset_epoch()
        train_metrics_adv.reset_epoch()
        val_metrics_clean.reset_epoch()
        val_metrics_adv.reset_epoch()
        print(f"Epoch {epoch+1}")
        #| eps: {current_eps:.4f}

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            if batch is None:
                continue

            imgs_raw, y, _ = batch
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
            # I need create_graph=True

            # Random start within epsilon ball
            delta = torch.empty_like(imgs_raw).uniform_(-current_eps, current_eps)
            imgs_perturbed = torch.clamp(imgs_raw + delta, 0, 1)

            # FGSM step from random start
            imgs_perturbed.requires_grad_(True)
            logits_perturbed = model(normalize(imgs_perturbed))
            loss_rs = criterion(logits_perturbed, y)

            #FGSM WITH RANDOM START
            grad_imgs = torch.autograd.grad(loss_rs, imgs_perturbed, create_graph=True)[0]
            imgs_adv = imgs_perturbed.detach() + current_eps * grad_imgs.detach().sign()
            imgs_adv = torch.clamp(imgs_raw + torch.clamp(imgs_adv - imgs_raw, -current_eps, current_eps), 0, 1)
            imgs_adv = normalize(imgs_adv.detach())
            imgs_adv = imgs_adv.detach().requires_grad_(True)

            # FGSM WITH NO RANDOM START
            #grad_imgs = torch.autograd.grad(loss_rs, imgs_perturbed, retain_graph=True, create_graph=True)[0]
            #imgs_adv = imgs_raw +  current_eps * grad_imgs.sign()
            #imgs_adv = torch.clamp(imgs_adv, 0, 1)
            ##imgs_adv = torch.clamp(imgs_adv, imgs - current_eps, imgs + current_eps)
            #imgs_adv = normalize(imgs_adv.detach())
            #imgs_adv.requires_grad_(True)
            
            # I compute again the clean loss with fresh graph
            #imgs_detached = imgs.detach()
            #imgs_detached.requires_grad = False

            #print(f"imgs_adv min/max: {imgs_adv.min():.3f}/{imgs_adv.max():.3f}")

            #Adversarial forward pass
            logits_adv = model(imgs_adv) #detach
            loss_adv = criterion(logits_adv, y)

            # gradient norm penalty
            # start with a small lambda_grad like 1e-4 or 1e-3
            loss_grad_clean = grad_imgs.view(imgs_raw.size(0), -1).norm(2, dim=1).mean()
            # GRADIENT NORM ON ADV EXAMPLES
            #grad_adv = torch.autograd.grad(loss_adv, imgs_adv, create_graph=True)[0]
            #loss_grad_adv = grad_adv.view(imgs_adv.size(0), -1).norm(2, dim=1).mean()

            if entropy_flag == True:
                # compute entropy
                entropy_clean = entropy_penalty(logits_clean)
                entropy_adv = entropy_penalty(logits_adv)
                # adding entropy penalty to the loss
                # In this version I'm applying entropy penalty only to the adversarial
                # examples, since I want to discourage overconfidence on them.
                # keep lambda entropy small, to
                # Then I apply also gradient penalty

                #+ lambda_grad * loss_grad_clean \

                loss = (1-alpha_adv) * loss_clean + alpha_adv * loss_adv \
                       + lambda_entropy * entropy_adv + lambda_grad * loss_grad_clean 
                       #+ lambda_grad * loss_grad_adv
            else:
                loss = (1-alpha_adv) * loss_clean + alpha_adv * loss_adv + lambda_grad * loss_grad_clean 
                #+ lambda_grad * loss_grad_adv 

            train_loss += loss.item() * imgs.size(0)
            epoch_loss = train_loss / len(train_loader.dataset)

            # Check everything before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skipping batch — NaN/Inf loss: {loss.item()}")
                optimizer.zero_grad()
                continue
            
            if torch.isnan(logits_clean).any() or torch.isnan(logits_adv).any():
                print(f"Skipping batch — NaN in logits")
                optimizer.zero_grad()
                continue

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Skip NaN batches
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

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

        pbar = tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
            if batch is None:
                continue
            imgs_raw, y, _ = batch
            imgs_raw, y = imgs_raw.to(device), y.to(device).long().squeeze()
            # Adversarial validation
            # Costruzione attacco - richiede il grafo
            with torch.enable_grad():
                imgs_raw = imgs_raw.detach().requires_grad_(True)
                imgs_norm = normalize(imgs_raw)
                logits_clean = model(imgs_norm)
                loss_tmp = criterion(logits_clean, y)
                grad_imgs = torch.autograd.grad(loss_tmp, imgs_raw)[0]

            imgs_adv = imgs_raw + 8/255 * grad_imgs.sign()
            imgs_adv = torch.clamp(imgs_adv, 0, 1)

            with torch.no_grad():
                logits_clean = model(normalize(imgs_raw.detach()))
                logits_adv = model(normalize(imgs_adv.detach()))
    
                probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_clean.update(y, probs_clean)
                probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_adv.update(y, probs_adv)

        val_results_clean = val_metrics_clean.compute()
        val_results_adv = val_metrics_adv.compute()

        val_acc_adv = val_metrics_adv.accuracy_list[epoch]
        clean_acc = val_metrics_clean.accuracy_list[epoch] 

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

        # early stopping
        #print(f"Epoch {epoch+1} - val_loss: {epoch_val_loss:.4f}")
        #if early_stopping(epoch_val_loss, model, optimizer, epoch, seed, eps_scheduler.type, "fgsm", train_metrics_clean, train_metrics_adv, val_metrics, train_losses, current_eps):
        #    break
        
    attack_name = ""
    if entropy_flag == True:
        attack_name = "square"
    else:
        attack_name = "fgsm"

    eps_sched_folder = ""
    save_path = ""
    history_path = ""
    
    #if adaptive == True:
    #    eps_sched_folder = "adaptive_curriculum_scheduler"
    #    save_path =  f'{MODELS_DIR}/pgn_train/resnet50_{attack_name}_epoch_{epoch}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_seed_{seed}_{eps_scheduler.type}_{lambda_grad}_pgn.pt'
    #    history_path = f"history/history_pgn/history_{attack_name}/{eps_sched_folder}/history_{attack_name}_epoch_{epoch}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_seed_{seed}_{eps_scheduler.type}_{lambda_grad}_pgn.json"
    #else:
    #    eps_sched_folder = "not_adaptive_curriculum_scheduler"
    #    save_path =  f'{MODELS_DIR}/{eps_sched_folder}/resnet50_{attack_name}_epoch_{epoch}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_seed_{seed}_{eps_scheduler.type}_pgn.pt'
    #    history_path = f"history/history_pgn/history_{attack_name}/{eps_sched_folder}/history_{attack_name}_epoch_{epoch}_LR_{LR}_batchsize_{BATCH_SIZE}_WD_{WD}_seed_{seed}_{eps_scheduler.type}_pgn.json"

    save_path =  f'{MODELS_DIR}/pgn_train/resnet50_{attack_name}_epoch_{epoch}_baselr_{base_lr}_maxlr_{max_lr}_alpha_adv_{alpha_adv}_batchsize_{BATCH_SIZE}_WD_{WD}_seed_{seed}_{eps_scheduler.type}_{lambda_grad}_pgn_clean_and_adv.pt'
    history_path = f"history/history_pgn/history_{attack_name}/history_{attack_name}_epoch_{epoch}_baselr_{base_lr}_maxlr_{max_lr}_alpha_adv_{alpha_adv}_batchsize_{BATCH_SIZE}_WD_{WD}_seed_{seed}_{eps_scheduler.type}_{lambda_grad}_pgn_clean_and_adv.json"

    if save_model == True:
        print(f"Saving models in {save_path}")
    
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
        },save_path)
        # save history
        print(f"Saving models in {history_path}")
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
            "train_epsilon": eps_scheduler.epsilons,
            "lambda_entropy": lambda_entropy,
            "lambda_grad": lambda_grad
        }
    
        save_history_json(history, history_path)
    

    return model, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    seed=42
    set_seed(42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])

    params = {"base_lr": 1e-06,
    "max_lr": 1e-04,
    "weight_decay": 0.01,
    "batch_size": 16,
    "step_size_up": 1,
    "lambda_entropy": 0.001,
    "num_epochs_per_eps": 1,
    "lambda_grad": 0.01
    }

    results     = []

    print("Initializing Data loaders ......")
    train_loader, val_loader, test_loader = get_data_loaders(transform, params['batch_size'])

    checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"

    eps_type = ['linear']

    for type in eps_type:
        print(f"Start training FGSM-AT + entropy penalty with {type} scheduler")
        model, criterion, optimizer, scheduler = reset_checkpoint(
            checkpoint_path=checkpoint_path, 
            sched_type="CyclicLR",
            base_lr=params['base_lr'], 
            max_lr=params['max_lr'], 
            wd=params['weight_decay'],
            step_size_up= params['num_epochs_per_eps'] * len(train_loader), 
            device=device)
        
        train_metrics_clean = Metrics()
        train_metrics_adv = Metrics()
        val_metrics_clean = Metrics()
        val_metrics_adv = Metrics()
        start_epoch = 0
        train_losses = []
    
        eps_scheduler = CurriculumEpsilonScheduler(
            eps_start=0/255, eps_end=8/255,
            num_epochs_rampup=30, type=type,
            adaptive=True, patience=5, num_epochs_per_eps=params['num_epochs_per_eps']
        )

        num_epochs = 30
        alpha_adv = 0.5
        entropy_flag = False
        training = ""
        attack = ""

        if entropy_flag == True:
            training = "FGSM-AT + entropy"
            attack = "square"
        else:
            training = "FGSM-AT"
            attack = "fgsm"

        
        # FGSM-AT + entropy penalty
        trained_model, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses = train_robust_with_curriculum_pgn(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        lambda_entropy=params['lambda_entropy'],
        lambda_grad=params['lambda_grad'],
        start_epoch=start_epoch, 
        num_epochs=num_epochs, 
        optimizer=optimizer,
        scheduler=scheduler, 
        eps_scheduler=eps_scheduler,
        criterion=criterion,
        device=device,
        train_losses=train_losses,
        train_metrics_clean=train_metrics_clean,
        train_metrics_adv=train_metrics_adv,
        val_metrics_clean=val_metrics_clean,
        val_metrics_adv=val_metrics_adv,
        seed=seed,
        base_lr=params["base_lr"],
        max_lr=params["max_lr"],
        alpha_adv=alpha_adv,
        entropy_flag=entropy_flag,
        adaptive=True,
        save_model=False)

        val_adv_acc = val_metrics_adv.accuracy_list[num_epochs-1]
        val_clean_acc = val_metrics_clean.accuracy_list[num_epochs-1]

        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

         # testing over FGSM
        clean_metrics, fgsm_metrics_1 = test_attack(trained_model, test_loader, 'fgsm', 2/255, 'foolbox', " ", " ", "FGSM (eps=2/255)", device, save_results=False)
        clean_metrics, fgsm_metrics_2 = test_attack(trained_model, test_loader, 'fgsm', 8/255, 'foolbox', " ", " ", "FGSM (eps=8/255)", device, save_results=False)
        clean_metrics, ifgsm_metrics_2 = test_attack(trained_model, test_loader, 'ifgsm', 8/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
        clean_metrics, pgd_metrics_2 = test_attack(trained_model, test_loader, 'pgd', 8/255, 'foolbox', " ", " ", "PGD", device, save_results=False)
        clean_metrics, ifgsm_metrics_1 = test_attack(trained_model, test_loader, 'ifgsm', 2/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
        clean_metrics, pgd_metrics_1 = test_attack(trained_model, test_loader, 'pgd', 2/255, 'foolbox', " ", " ", "PGD", device, save_results=False)
    
    
        if torch.isnan(torch.tensor(val_clean_acc)):
            results.append({
                'params': params, 
                'val_clean_acc': None, 
                'val_adv_acc':  None,
                'test_clean_acc':  None,
                'test_fgsm_small_acc':  None,
                'test_fgsm_big_acc':  None,
                'test_ifgsm_big_acc':  None,
                'test_pgd_big_acc':  None,
                'status': 'failed_nan'})
        else:
            results.append({
                'params': params, 
                'val_clean_acc': val_clean_acc, 
                'val_adv_acc': val_adv_acc,
                'test_clean_acc': clean_metrics.accuracy_list[0],
                'test_fgsm_small_acc': fgsm_metrics_1.accuracy_list[0],
                'test_fgsm_big_acc': fgsm_metrics_2.accuracy_list[0],
                'test_ifgsm_small_acc': ifgsm_metrics_1.accuracy_list[0],
                'test_ifgsm_big_acc': ifgsm_metrics_2.accuracy_list[0],
                'test_pgd_small_acc': pgd_metrics_1.accuracy_list[0],
                'test_pgd_big_acc': pgd_metrics_2.accuracy_list[0],
                'alpha_adv': alpha_adv,
                'training': training + "_random_start",
                'pgn': 'clean',
                'status': 'ok'})
        
        with open(f'results_grid_search/linear_per_epoch/grid_search_{attack}_pgn_with_test_results_uniform_Cyclic_{num_epochs}_alpha_{alpha_adv}.json', 'w') as f:
            json.dump(results, f, indent=4)
    
        print(f"Clean acc: {val_clean_acc:.4f} | Adv acc: {val_adv_acc:.4f}")

        

