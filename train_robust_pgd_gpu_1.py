import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import *
import foolbox as fb
from test_attacks import test_attack
from ades_difat.ades import LearnableEpsilonScheduler, compute_adaptive_epsilon, pgd_attack_adaptive_eps
import itertools


# PGD-AT with epsilon scheduler and gradient norm penalty
def train_robust_PGD(model, train_loader, val_loader, lambda_entropy, lambda_grad, lambda_fgsm,
                      start_epoch, num_epochs, optimizer, scheduler, LRscheduler_type, eps_scheduler, 
                      criterion, device, train_losses, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv,
                        seed, base_lr, max_lr, alpha_adv, mode='baseline', ades_scheduler=None, ades_optimizer=None,
                        ades_eps_min=2/255, ades_eps_lambda=6/255, ades_mc_passes=3,
                        epsilon=2/255, entropy_flag=False, has_eps_scheduler=False, has_gradient_penalty=False, has_fgsm_aux_loss=False, save_model=True):
    assert mode in ("baseline", "ades", "difat"), f"unknown mode: {mode}"
    if mode == "ades":
        assert ades_scheduler is not None and ades_optimizer is not None, \
            "mode='ades' requires an EpsilonScheduler + its own optimizer"
    
    train_loss = 0.0
    history = {}
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    val_acc_adv = 0
    clean_acc   = 0

    # Initializing pgd attack from foolbox library
    attack_pgd = fb.attacks.LinfPGD(
        steps=20,
        #rel_stepsize=0.25, # it is used to calculate alpha = rel_stepsize * epsilon
        abs_stepsize=1/255,
        random_start=True)
    
    eval_pgd = fb.attacks.LinfPGD(
        steps=20,
        #rel_stepsize=0.25,
        abs_stepsize=1/255,
        random_start=True)
    
    fgsm = fb.attacks.FGSM()
    
    preprocessing = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    axis=-3  # per PyTorch (C, H, W)
    )
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing, device=device)
    print(f"Training PGD-AT | mode={mode} | eps={epsilon:.4f}")

    for epoch in range(start_epoch, num_epochs):
        #TRAINING
        model.train()
        train_loss = 0.0

        if mode == "baseline":

            if eps_scheduler.should_stop():
                current_eps = 8/255
            else:
                current_eps = eps_scheduler.get_epsilon(val_acc_adv, clean_acc, epoch)

        else:
            current_eps = epsilon

        # I block update statistics of BatchNorm
        freeze_bn(model)
        train_metrics_clean.reset_epoch()
        train_metrics_adv.reset_epoch()
        val_metrics_clean.reset_epoch()
        val_metrics_adv.reset_epoch()
        print(f"Epoch {epoch+1}")

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            if batch is None:
                continue

            imgs_raw, y, _ = batch
            #imgs_raw, y = imgs_raw.to(device), y.to(device).long().squeeze()
            imgs_raw, y = imgs_raw.to(device), y.to(device).long().view(-1)
            #print(f"imgs_raw min/max: {imgs_raw.min():.3f}/{imgs_raw.max():.3f}")
            #print(f"imgs min/max: {imgs.min():.3f}/{imgs.max():.3f}")
            assert y.dim() == 1 and y.shape[0] == imgs_raw.shape[0], f"bad y shape: {y.shape}"
            imgs_foolbox =  imgs_raw.clone()

            # clean forward
            optimizer.zero_grad()

            imgs_raw.requires_grad_(True)
            imgs = normalize(imgs_raw)
            logits_clean = model(imgs)
            loss_clean = criterion(logits_clean, y)
            grad_clean = torch.autograd.grad(loss_clean, imgs_raw, create_graph=True)[0]

            # compute adversarial examples

            if mode == "baseline":
                _, imgs_adv, _ = attack_pgd(fmodel, imgs_foolbox, y, epsilons=current_eps)
                imgs_adv = normalize(imgs_adv).detach().requires_grad_(True)
            elif mode == "ades":
                eps_x, sigma = compute_adaptive_epsilon(
                        ades_scheduler, model, imgs_raw.detach(), y, criterion, normalize,
                        eps_min=ades_eps_min, eps_lambda=ades_eps_lambda, mc_passes=ades_mc_passes,
                    )
                print(f"ADES EPSILON: {eps_x}")
                _, imgs_adv, _ = attack_pgd(fmodel, imgs_foolbox, y, epsilons=eps_x)

            imgs_adv = normalize(imgs_adv).detach().requires_grad_(True)

            optimizer.zero_grad()


            # clean gradient penalty
            loss_grad_clean = grad_clean.view(imgs_raw.size(0), -1).norm(2, dim=1).mean()

            # Adversarial forward
            logits_adv = model(imgs_adv)
            loss_adv = criterion(logits_adv, y)

            if entropy_flag == True:
                # compute entropy
                #entropy_clean = entropy_penalty(logits_clean)
                entropy_adv = entropy_penalty(logits_adv)
                # adding entropy penalty to the loss
                # In this version I'm applying entropy penalty only to the adversarial
                # examples, since I want to discourage overconfidence on them.
                # keep lambda entropy small, to
                # Then I apply also gradient penalty

                if has_gradient_penalty == True:
                    loss = (1-alpha_adv) * loss_clean + alpha_adv * loss_adv + lambda_entropy * entropy_adv \
                           + lambda_grad * loss_grad_clean 
                           #+ lambda_grad * loss_grad_adv
                else:
                    loss = (1-alpha_adv) * loss_clean + alpha_adv * loss_adv + lambda_entropy * entropy_adv
                           #+ lambda_grad * loss_grad_clean 
                           #+ lambda_grad * loss_grad_adv
            else:

                if has_gradient_penalty == True:
                    loss = (1-alpha_adv) * loss_clean + alpha_adv * loss_adv \
                    + lambda_grad * loss_grad_clean 
                    #+ lambda_grad * loss_grad_adv 
                else:
                    loss = (1-alpha_adv) * loss_clean + alpha_adv * loss_adv 
                    #+ lambda_grad * loss_grad_clean 
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
            if LRscheduler_type == "CyclicLR":
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
            imgs_raw, y = imgs_raw.to(device), y.to(device).long().view(-1)
            imgs_foolbox =  imgs_raw.detach().clone()
            assert y.dim() == 1 and y.shape[0] == imgs_raw.shape[0], f"bad y shape: {y.shape}"

            _, imgs_adv, _ = eval_pgd(fmodel, imgs_foolbox, y, epsilons=8/255)

            with torch.no_grad():
                logits_clean = model(normalize(imgs_raw))
                logits_adv = model(normalize(imgs_adv))
    
                probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_clean.update(y, probs_clean)
                probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_adv.update(y, probs_adv)

        val_results_clean = val_metrics_clean.compute()
        val_results_adv = val_metrics_adv.compute()

        val_acc_adv = val_metrics_adv.accuracy_list[epoch]
        clean_acc = val_metrics_clean.accuracy_list[epoch] 

        if LRscheduler_type == "MultiStepLR":
            scheduler.step()

        print(f"Epoch {epoch+1}:")
        print("TRAINING")
        print("Training loss:", epoch_loss)
        print("CLEAN RESULTS")
        train_metrics_clean.print(epoch)
        print("PGD RESULTS")
        train_metrics_adv.print(epoch)
        print("ATTACK SUCCESS RATE")
        print(train_metrics_adv.asr_list[epoch])

        print("CLEAN VALIDATION")
        val_metrics_clean.print(epoch)
        print("ADV VALIDATION")
        val_metrics_adv.print(epoch)
        
    if entropy_flag == False:
        attack_name = "pgd20"
    else:
        attack_name = f"pgd_entropy_{lambda_entropy}"

    if has_eps_scheduler == True:
        eps_label = f"eps_sched_{eps_scheduler.type}"
    else:
        eps_label = f"epsilon_{current_eps}"

    grad_penalty = "_"
    if has_gradient_penalty == True:
        grad_penalty = f"_with_pgn_{lambda_grad}"

    fgsm_aux_loss_label = "_"
    if has_fgsm_aux_loss == True:
        fgsm_aux_loss_label = f"_with_fgsm_aux_{lambda_fgsm}"

    

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

    save_path =  f'{MODELS_DIR}/pgd_train/resnet50_{attack_name}{grad_penalty}_epoch_{epoch}_{epoch/2}_epochsrampup_baselr_{base_lr}_maxlr_{max_lr}_alpha_adv_{alpha_adv}_batchsize_{BATCH_SIZE}_WD_{WD}_{eps_label}_{fgsm_aux_loss_label}_seed_{seed}.pt'
    history_path = f"history/history_{attack_name}/history_{attack_name}{grad_penalty}_epoch_{epoch}_{epoch/2}_epochsrampup_baselr_{base_lr}_maxlr_{max_lr}_alpha_adv_{alpha_adv}_batchsize_{BATCH_SIZE}_WD_{WD}_{eps_label}_{fgsm_aux_loss_label}_seed_{seed}.json"

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
        "train_epsilon": eps_scheduler.type,
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

    # WD SGD = 5e-4, LR = 1e-3
    # Ciclyc lr = 1e-5, wd = 1e-3

    param_grid = {"base_lr": [1e-03],
    "max_lr": [1e-05],
    "weight_decay": [5e-4],
    "batch_size": [16],
    "step_size_up": [3],
    #"lambda_entropy": [0.001, 0.005, 0.01],
    "lambda_entropy": [0.01],
    "num_epochs_per_eps": [1],
    #"lambda_grad": [0.1, 0.05, 0.01]
    "lambda_grad": [0.1]
    }

    # PARAMETERS #################
    num_epochs = 15
    alpha_adv = 0.5
    epsilon = 8/255
    entropy_flag = False
    has_eps_scheduler = False
    has_gradient_penalty = False
    has_fgsm_aux_loss = False
    lambda_fgsm = 0

    # Generate all combinations
    keys   = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in itertools.product(*values):

        params = dict(zip(keys, combo))

        print(f"\nTesting: {params}")
    
        torch.cuda.empty_cache()
    
        results     = []
    
        print("Initializing Data loaders ......")
        train_loader, val_loader, test_loader = get_data_loaders(transform, params['batch_size'])
    
        checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
        sched_type = "MultiStepLR"
        optimizer_type = "SGD"
    
        print(f"Start training PGD-AT")
        model, criterion, optimizer, scheduler = reset_checkpoint(
            num_epochs=num_epochs,
            checkpoint_path=checkpoint_path, 
            optimizer_type=optimizer_type,
            sched_type=sched_type,
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
        epsilon_scheduler = None
        ades_scheduler = None
        ades_optimizer = None
        difat_purifier = None

        # TRAINING SETTINGS
        mode = "ades"

        if mode == "ades":
            print("ADES CHECK MAIN")
            ades_scheduler = LearnableEpsilonScheduler()
            ades_scheduler = ades_scheduler.to(device)
            ades_optimizer = optimizer = torch.optim.AdamW(ades_scheduler.parameters(), weight_decay=1e-5)
    
        # EPSILON SCHEDULER
        type = 'linear'
        eps_scheduler = CurriculumEpsilonScheduler(
                eps_start=0/255, eps_end=8/255,
                num_epochs_rampup=int(num_epochs/2), type=type,
                adaptive=True, patience=5, num_epochs_per_eps=params['num_epochs_per_eps']
            )
        ##############################
    
        training = ""
        attack = ""
        if entropy_flag == True:
            training = "PGD-AT + entropy"
            attack = f"pgd_entropy_{params['lambda_entropy']}"
        else:
            training = "PGD-AT"
            attack = "pgd"
    
        if has_eps_scheduler == True:
            eps_label = f"eps_sched_{eps_scheduler.type}"
        else:
            eps_label = f"epsilon_{epsilon}"
    
        grad_penalty = "_"
        if has_gradient_penalty == True:
            grad_penalty = f"_with_pgn_lambda_{params['lambda_grad']}"
    
        fgsm_aux_loss = "_"
        if has_fgsm_aux_loss == True:
            fgsm_aux_loss = f"_with_fgsm_aux_{lambda_fgsm}"
    
        
        # PGD_AT
        print(f"has_gradient_penalty: {has_gradient_penalty}, {params['lambda_grad']}")
        
        trained_model, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses = train_robust_PGD(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        lambda_entropy=params["lambda_entropy"],
        lambda_grad=params["lambda_grad"],
        lambda_fgsm = lambda_fgsm,
        start_epoch=start_epoch, 
        num_epochs=num_epochs, 
        optimizer=optimizer,
        scheduler=scheduler, 
        LRscheduler_type=sched_type,
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
        mode="ades",
        ades_scheduler=ades_scheduler,
        ades_optimizer=ades_optimizer,
        epsilon=epsilon,
        entropy_flag=entropy_flag,
        has_eps_scheduler=has_eps_scheduler,
        has_gradient_penalty=has_gradient_penalty,
        has_fgsm_aux_loss = has_fgsm_aux_loss,
        save_model=False)
    
        val_adv_acc = val_metrics_adv.accuracy_list[num_epochs-1]
        val_clean_acc = val_metrics_clean.accuracy_list[num_epochs-1]
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
         # testing
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
                'test_clean_auc': clean_metrics.auc_list[0],
                'test_clean_asr': 0,
                'test_fgsm_small_acc': fgsm_metrics_1.accuracy_list[0],
                'test_fgsm_small_auc': fgsm_metrics_1.auc_list[0],
                'test_fgsm_small_asr': fgsm_metrics_1.asr_list[0],
                'test_fgsm_big_acc': fgsm_metrics_2.accuracy_list[0],
                'test_fgsm_big_auc': fgsm_metrics_2.auc_list[0],
                'test_fgsm_big_asr': fgsm_metrics_2.asr_list[0],
                'test_ifgsm_small_acc': ifgsm_metrics_1.accuracy_list[0],
                'test_ifgsm_small_auc': ifgsm_metrics_1.auc_list[0],
                'test_ifgsm_small_asr': ifgsm_metrics_1.asr_list[0],
                'test_ifgsm_big_acc': ifgsm_metrics_2.accuracy_list[0],
                'test_ifgsm_big_auc': ifgsm_metrics_2.auc_list[0],
                'test_ifgsm_big_asr': ifgsm_metrics_2.asr_list[0],
                'test_pgd_small_acc': pgd_metrics_1.accuracy_list[0],
                'test_pgd_small_auc': pgd_metrics_1.auc_list[0],
                'test_pgd_small_asr': pgd_metrics_1.asr_list[0],
                'test_pgd_big_acc': pgd_metrics_2.accuracy_list[0],
                'test_pgd_big_auc': pgd_metrics_2.auc_list[0],
                'test_pgd_big_asr': pgd_metrics_2.asr_list[0],
                'alpha_adv': alpha_adv,
                'eps_scheduler': eps_scheduler.type,
                'training': training,
                'pgn': f'{grad_penalty}',
                'fgsm_aux_loss': f'{fgsm_aux_loss}',
                'status': 'ok'})
            
        out_dir = f'pgd_training_{mode}'
        os.makedirs(out_dir, exist_ok=True)
        
        with open(f'results_grid_search/{out_dir}/pgd_training_{mode}{grad_penalty}_{eps_label}_with_test_results_{sched_type}_{num_epochs}_{num_epochs/2}_epochsrampup_alpha_{alpha_adv}.json', 'w') as f:
            json.dump(results, f, indent=4)
    
        print(f"Clean acc: {val_clean_acc:.4f} | Adv acc: {val_adv_acc:.4f}")


    