"""
PGD-AT training loop supporting three attack modes:
  - "baseline": standard fixed-epsilon PGD-AT
  - "ades":     ADES learnable per-sample epsilon scheduler (pgd_at_ades.py)
  - "difat":    DPGD + diffusion purification (pgd_at_difat.py)
 
Kept as ONE shared loop (rather than three duplicated ~300-line copies) so the
metrics/timing/history-saving conventions from your existing pipeline stay in
exactly one place. The two mechanisms remain independently usable: pass
mode="ades" or mode="difat" to run either alone; a "combined" mode is a
straightforward extension once you've validated each independently (swap the
adaptive-eps output from ADES into dpgd_attack's `eps` argument per-sample).
"""
import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
from torchvision import transforms
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from utils import *
import itertools
from test_attacks import test_attack
from diffusers import DDPMPipeline, DDPMScheduler
 
from ades import LearnableEpsilonScheduler, compute_adaptive_epsilon, pgd_attack_adaptive_eps, pgd_attack_adaptive_eps_differentiable
from difat import DiffusionPurifier, dpgd_attack
 
 
def standard_pgd_attack(model, x, y, eps, alpha, steps, normalize, clamp_min=0.0, clamp_max=1.0):
    """Your existing fixed-epsilon PGD-AT attack, as the baseline mode."""
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = torch.clamp(x + delta, clamp_min, clamp_max) - x
    delta = delta.detach().requires_grad_(True)
 
    for _ in range(steps):
        logits = model(normalize(x + delta))
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = delta.detach() + alpha * grad.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(x + delta, clamp_min, clamp_max) - x
        delta.requires_grad_(True)
 
    return torch.clamp(x + delta.detach(), clamp_min, clamp_max)

def linear_scheduler(num_epochs, current_epoch):
    if current_epoch < num_epochs // 2:
        target_eps = (current_epoch / (num_epochs // 2)) * (8/255)
    else:
        target_eps = 8/255
    return target_eps

 
def train_pgd_at(model, train_loader, val_loader, start_epoch, num_epochs, optimizer, LRscheduler, criterion, device, train_losses, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, 
                  eps=8/255, alpha=2/255, steps=8, lr=1e-3,
                  mode="baseline", epsilon_scheduler=None,    # "baseline" | "ades" | "difat"
                  # --- ADES-specific ---
                  ades_scheduler=None, ades_optimizer=None, ades_eps_min=0/255,
                  ades_eps_lambda=6/255, ades_mc_passes=3, lambda_ades = 0.01, lambda_mean = 1.0, beta = 0.01,
                  # --- DifAT-specific ---
                  difat_purifier=None, difat_margin_c=1.0, difat_tau=1,
                  seed=0, save_model=True):
 
    
    if mode == "difat":
        assert difat_purifier is not None, "mode='difat' requires a DiffusionPurifier"
 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_acc_adv = 0
    clean_acc   = 0
    target_eps = (ades_eps_min + 8/255) / 2
    margin = 0.5 / 255 # I allow the average to get close to eps_min
    loss_type = "None"
    history = {}

    eps_stats = {
    "mean": [],
    "std": [],
    "min": [],
    "max": [],
    "median": [],
    }
 
    epoch_train_times, epoch_val_times, epoch_total_times = [], [], []
    training_start_time = time.perf_counter()
 
    print(f"Training PGD-AT | mode={mode} | eps={eps:.4f} alpha={alpha:.4f} steps={steps}")
 
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.perf_counter()
        model.train()
        train_loss, train_correct_clean, train_correct_adv, n_seen = 0.0, 0, 0, 0
        epoch_eps = []

        if mode in ("baseline","difat"):
            if epsilon_scheduler != None:
                if epsilon_scheduler.should_stop():
                        eps = 8/255
                else:
                    eps = epsilon_scheduler.get_epsilon(val_acc_adv, clean_acc, epoch)
            else:
                eps = eps

        freeze_bn(model)
        train_metrics_clean.reset_epoch()
        train_metrics_adv.reset_epoch()
        val_metrics_clean.reset_epoch()
        val_metrics_adv.reset_epoch()
                
 
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            if batch is None:
                continue
            imgs_raw, y, _ = batch
            imgs_raw, y = imgs_raw.to(device), y.to(device).long().view(-1)
            assert y.dim() == 1, f"y must be 1-d, got shape {y.shape}"
 
            optimizer.zero_grad()
            if mode == "ades":
                ades_optimizer.zero_grad()
 
            imgs = normalize(imgs_raw)
            logits_clean = model(imgs)
            if torch.isnan(logits_clean).any() or torch.isinf(logits_clean).any():
                print("[WARNING] NaN/Inf in clean logits, skipping")
                continue
            loss_clean = criterion(logits_clean, y)
 
            # ------------------------------------------------------------------
            # Attack generation, mode-dependent
            # ------------------------------------------------------------------
            if mode == "baseline":
                #print(f"epsilon:{eps}")
                imgs_adv_raw = standard_pgd_attack(model, imgs_raw.detach(), y, eps, alpha, steps, normalize)
                imgs_adv = normalize(imgs_adv_raw.detach().requires_grad_(True)) # detach() or not detach()???
                sigma = None
 
            elif mode == "ades":
                eps_x, sigma = compute_adaptive_epsilon(
                        ades_scheduler, model, imgs_raw.detach(), y, criterion, normalize,
                        eps_min=ades_eps_min, eps_lambda=ades_eps_lambda, mc_passes=ades_mc_passes,
                    ) #imgs_raw.detach()
                epoch_eps.append(eps_x.detach().cpu())
                mean_eps = eps_x.mean()
                #print(eps_x)
                #print("eps_x.requires_grad:", eps_x.requires_grad)
                #print("eps_x.grad_fn:", eps_x.grad_fn)
                #eps_x.retain_grad()
                # compute_adaptive_epsilon runs estimate_mc_uncertainty that calls model.eval()
                # so I need to restore the training mode
                #print("training mode:", model.training)
                model.train()
                #print("training mode:", model.training)
                #print(f"ADES EPSILON: {eps_x}")
                imgs_adv_raw = pgd_attack_adaptive_eps_differentiable(
                    model, normalize, imgs_raw.detach(), y, eps_x, alpha, steps
                )
                imgs_adv = normalize(imgs_adv_raw)
                #print(f"imgs_adv_raw.grad_fn: {imgs_adv_raw.grad_fn}")
 
            elif mode == "difat":
                imgs_adv_raw = dpgd_attack(
                    model, imgs_raw.detach(), y, eps, alpha, steps,
                    purifier=difat_purifier, margin_c=difat_margin_c,
                    control_factor_tau=difat_tau,
                )
                sigma = None
                imgs_adv = normalize(imgs_adv_raw.detach().requires_grad_(True))
 
            logits_adv = model(imgs_adv)
            if torch.isnan(logits_adv).any() or torch.isinf(logits_adv).any():
                print("[WARNING] NaN/Inf in adversarial logits, skipping")
                continue
            loss_adv = criterion(logits_adv, y)
 
            loss = 0.5 * loss_clean + 0.5 * loss_adv
 
            # ADES scheduler is trained jointly, through the adaptive-eps path.
            # sigma only carries gradient if compute_adaptive_epsilon kept the
            # graph -- attach a light auxiliary term so the scheduler receives
            # a training signal (encourage larger eps on samples that end up
            # harder, i.e. correlate sigma with per-sample adversarial loss).
            if mode == "ades" and sigma is not None:
                with torch.no_grad():
                    per_sample_adv_loss = F.cross_entropy(logits_adv, y, reduction="none")
                    target_sigma = (per_sample_adv_loss / (per_sample_adv_loss.max() + 1e-8))
                # SCHEDULER LOSS 1
                #loss_type = "LOSS1"
                #scheduler_loss = F.mse_loss(sigma, target_sigma)
                #loss = loss + lambda_ades * scheduler_loss

                # SCHEDULER LOSS 2
                #loss_type = "LOSS2"
                #scheduler_loss = F.relu((ades_eps_min + margin) - avg_eps) ** 2
                #loss = loss + lambda_ades * scheduler_loss

                # SCHEDULER LOSS 3
                loss_type = "MAXLOSS_LINEAR_TARGET"
                #scheduler_loss = -loss_adv - beta * eps_x.var(unbiased=False)
                target_eps = linear_scheduler(num_epochs, epoch) / (8/255)
                current_mean = mean_eps / (8/255)
                adv_term = -loss_adv
                mean_term = lambda_mean * (current_mean - target_eps).pow(2)
                var_term = - beta * eps_x.var(unbiased=False)
                scheduler_loss = (
                   adv_term
                   + mean_term
                   + var_term
               )

                #print(
                #    f"adv={adv_term.item():.3f} "
                #    f"mean={mean_term.item():.3f} "
                #    f"var={var_term.item():.3f}"
                #)
 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item() * imgs.size(0)
            epoch_loss = train_loss / len(train_loader.dataset)

            if mode == "ades":
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            #print("eps_x.grad:", eps_x.grad)
            #print("sigma.grad =", sigma.grad)
            #if mode == "ades":
            #    for name, p in ades_scheduler.named_parameters():
            #        if p.grad is None:
            #            print(name, "None")
            #        else:
            #            print(name, p.grad.norm().item())   

            if mode == "ades":
                # Freeze detector
                for p in model.parameters():
                    p.requires_grad = False
                scheduler_loss.backward()
                # Unfreeze detector
                for p in model.parameters():
                    p.requires_grad = True

            optimizer.step()
            if mode == "ades":
                ades_optimizer.step()


            #for name, p in ades_scheduler.named_parameters():
            #    if p.grad is None:
            #        print(name, "grad = None")
            #    else:
            #        print(name, p.grad.norm().item())
 
            #train_loss += loss.item() * imgs.size(0)
            #print(loss_clean.item(), loss_adv.item())
            n_seen += imgs.size(0)
            train_correct_clean += (logits_clean.argmax(1) == y).sum().item()
            train_correct_adv += (logits_adv.argmax(1) == y).sum().item()

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            train_metrics_clean.update(y, probs_clean)
            probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
            train_metrics_adv.update(y, probs_adv)
 
        #epoch_loss = train_loss / max(n_seen, 1)
        train_losses.append(epoch_loss)
        train_results_clean = train_metrics_clean.compute()
        train_results_adv = train_metrics_adv.compute()
        train_metrics_adv.attack_success_rate(train_metrics_clean.all_probs)
        # I save the avg epsilon chosen by the ADES scheduler
        if mode == "ades":
            epoch_eps = torch.cat(epoch_eps)
            #eps_epoch_mean = statistics.mean(avg_eps_per_batch)
            #print(eps_epoch_mean)
            #avg_eps_per_epoch.append(eps_epoch_mean)
            #avg_eps_per_batch = []
            eps_stats["mean"].append(epoch_eps.mean().item())
            eps_stats["std"].append(epoch_eps.std().item())
            eps_stats["min"].append(epoch_eps.min().item())
            eps_stats["max"].append(epoch_eps.max().item())
            eps_stats["median"].append(epoch_eps.median().item())
            
            print(eps_stats)

 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_train_time = time.perf_counter() - epoch_start_time
        epoch_train_times.append(epoch_train_time)
 
        # ---------------------------------------------------------------------
        # Validation (plain PGD, no purification/adaptive-eps, for fair
        # cross-mode comparison of final robustness)
        # ---------------------------------------------------------------------
        val_start_time = time.perf_counter()
        model.eval()
        val_loss, val_correct_clean, val_correct_adv, n_val = 0.0, 0, 0, 0
 
        for batch in tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            imgs_raw, y, _ = batch
            imgs_raw, y = imgs_raw.to(device), y.to(device).long().view(-1)
 
            with torch.enable_grad():
                imgs_adv_raw = standard_pgd_attack(model, imgs_raw.detach(), y, 8/255, 1/255, 20, normalize)
 
            with torch.no_grad():
                logits_clean = model(normalize(imgs_raw))
                logits_adv = model(normalize(imgs_adv_raw))
                loss_clean = criterion(logits_clean, y)
                loss_adv = criterion(logits_adv, y)
                loss = 0.5 * loss_clean + 0.5 * loss_adv
 
                val_loss += loss.item() * imgs_raw.size(0)
                n_val += imgs_raw.size(0)
                val_correct_clean += (logits_clean.argmax(1) == y).sum().item()
                val_correct_adv += (logits_adv.argmax(1) == y).sum().item()

                probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_clean.update(y, probs_clean)
                probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_adv.update(y, probs_adv)

        val_results_clean = val_metrics_clean.compute()
        val_results_adv = val_metrics_adv.compute()

        val_acc_adv = val_metrics_adv.accuracy_list[epoch]
        clean_acc = val_metrics_clean.accuracy_list[epoch] 
 
        epoch_val_loss = val_loss / max(n_val, 1)

        LRscheduler.step()
 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_val_time = time.perf_counter() - val_start_time
        epoch_val_times.append(epoch_val_time)
        epoch_total_times.append(epoch_train_time + epoch_val_time)
 
        print(f"Epoch {epoch+1} [{mode}] | train_loss={epoch_loss:.4f} val_loss={epoch_val_loss:.4f} | "
              f"train_clean_acc={train_correct_clean/max(n_seen,1):.4f} "
              f"train_adv_acc={train_correct_adv/max(n_seen,1):.4f} | "
              f"val_clean_acc={val_correct_clean/max(n_val,1):.4f} "
              f"val_adv_acc={val_correct_adv/max(n_val,1):.4f}")
        print(f"[TIMING] train={epoch_train_time:.2f}s val={epoch_val_time:.2f}s "
              f"cumulative={time.perf_counter()-training_start_time:.2f}s")
 
    total_time = time.perf_counter() - training_start_time
    print(f"[TIMING] Total: {total_time:.2f}s ({total_time/60:.2f} min) over {num_epochs} epochs, "
          f"mode={mode}, avg {sum(epoch_total_times)/max(len(epoch_total_times),1):.2f}s/epoch")

    if mode == "ades":
        mode = f"ades_{loss_type}_lambda_mean_{lambda_mean}"

    epsilon_label = ""
    if mode == "baseline":
        epsilon_label = f"_linear_eps_sched_numeprampup{int(num_epochs/2)}"
    
    save_path = f'{MODELS_DIR}/pgdat_ades_difat/resnet50_pgdat_{mode}_{epsilon_label}_lr_{lr}_seed_{seed}_epochs_{num_epochs}_freeze.pt'
    history_path = f"history/history_pgdat_ades_difat/history_pgdat_{mode}_{epsilon_label}_lr_{lr}_seed_{seed}_epochs_{num_epochs}_freeze.json"
 
    if save_model:
        torch.save({
            "mode": mode, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "epoch_train_times_sec": epoch_train_times,
            "epoch_val_times_sec": epoch_val_times,
            "epoch_total_times_sec": epoch_total_times,
            "total_training_time_sec": total_time,
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
        }, save_path)
        print(f"Saved model to {save_path}")

    print(f"Saving models in {history_path}")
    history = {
            "epoch": epoch+1,
            "alpha": alpha,
            "steps": steps,
            "lambda_mean": lambda_mean,
            "beta": beta,
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
            "eps_stats": eps_stats
        }
    
    save_history_json(history, history_path)
    

    return model, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses, epoch_total_times, loss_type

 
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    seed=42
    set_seed(42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])

    params = {"lr": 1e-03,
    "weight_decay": 5e-4,
    "batch_size": 16
    }

    mode_list = ["ades"]

    # Generate all combinations
    #keys   = list(param_grid.keys())
    #values = list(param_grid.values())

    for mode in mode_list:

        #params = dict(zip(keys, combo))
#
        #print(f"\nTesting: {params}")
    #
        #torch.cuda.empty_cache()
    
        results     = []
    
        print("Initializing Data loaders ......")
        train_loader, val_loader, test_loader = get_data_loaders(transform, params['batch_size'])
    
        checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
        #mode = 'baseline'
        epsilon_scheduler = None
        ades_scheduler = None
        ades_optimizer = None
        difat_purifier = None
        lambda_ades = 0
        lambda_mean = 8.0
        beta = 0.01
        train_metrics_clean = Metrics()
        train_metrics_adv = Metrics()
        val_metrics_clean = Metrics()
        val_metrics_adv = Metrics()
        start_epoch = 0
        train_losses = []
        num_epochs = 25
        alpha_adv = 0.5
        epsilon = 8/255
    
        print(f"Start training PGD-AT with mode {mode}")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # I modify the last layer for binary classification
        model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(model.fc.in_features, 2)
        )
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        criterion = nn.CrossEntropyLoss()
        lr = 1e-4
        #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        # OPTIMIZER AND LR SCHEDULER RESNET50
        optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params['lr'],              # your empirically-successful ceiling was ~1e-5 under CyclicLR;
                               # SGD+momentum can typically tolerate a somewhat higher peak LR
                               # than Adam-family optimizers for the same model, but I'd still
                               # start conservatively and treat this as a value to sweep
        momentum=0.9,
        weight_decay=5e-4,    # matches the paper directly, no scaling needed here
        )
        # paper: decay at epochs 75/90 out of 100 → i.e. at 75% and 90% of total training
        milestones = [int(0.75 * num_epochs), int(0.90 * num_epochs)]  # e.g. [15, 18] for your 20 epochs
        LRscheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        # LEARNABLE ADES EPSILON SCHEDULER
        if mode == "ades":
            print("ADES CHECK MAIN")
            ades_scheduler = LearnableEpsilonScheduler()
            ades_scheduler = ades_scheduler.to(device)
            ades_optimizer = torch.optim.AdamW(ades_scheduler.parameters(), weight_decay=1e-5)
            lambda_ades = 0.001

        #DIFAT
        if mode == "difat":
            pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
            difat_purifier = DiffusionPurifier(pipe.unet, pipe.scheduler, device=device)

        #Baseline with linear epsilon scheduler
        # EPSILON SCHEDULER
        #if mode == "baseline":
        type_sched = 'linear'
        num_epochs_rampup = 12
        epsilon_scheduler = CurriculumEpsilonScheduler(
                eps_start=0/255, eps_end=8/255,
                num_epochs_rampup=num_epochs_rampup, type=type_sched,
                adaptive=True, patience=5, num_epochs_per_eps=1
            )
        ##############################



        print(f"Starting training PGD-AT with mode: {mode}")
        
        trained_model, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses, epoch_total_times, loss_type = train_pgd_at(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch,
            num_epochs=num_epochs,
            optimizer=optimizer,
            LRscheduler=LRscheduler,
            criterion=criterion,
            device=device,
            train_losses=train_losses,
            train_metrics_clean=train_metrics_clean,
            train_metrics_adv=train_metrics_adv,
            val_metrics_clean=val_metrics_clean,
            val_metrics_adv=val_metrics_adv,
            lr=params["lr"],
            mode=mode,
            epsilon_scheduler=epsilon_scheduler,
            ades_scheduler=ades_scheduler,
            ades_optimizer=ades_optimizer,
            lambda_ades=lambda_ades,
            lambda_mean=lambda_mean,
            beta =beta,
            seed=seed,
            difat_purifier=difat_purifier,
            save_model=True
        )
    
        val_adv_acc = val_metrics_adv.accuracy_list[num_epochs-1]
        val_clean_acc = val_metrics_clean.accuracy_list[num_epochs-1]
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
         # testing
        clean_metrics, fgsm_metrics_1 = test_attack(trained_model, test_loader, 'fgsm', 2/255, 'foolbox', " ", " ", "FGSM (eps=2/255)", device, save_results=False)
        clean_metrics, fgsm_metrics_2 = test_attack(trained_model, test_loader, 'fgsm', 8/255, 'foolbox', " ", " ", "FGSM (eps=8/255)", device, save_results=False)
        clean_metrics, ifgsm_metrics_2 = test_attack(trained_model, test_loader, 'ifgsm', 8/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
        clean_metrics, pgd_metrics_2 = test_attack(trained_model, test_loader, 'pgd', 8/255, 'None', " ", " ", "PGD", device, save_results=False)
        clean_metrics, ifgsm_metrics_1 = test_attack(trained_model, test_loader, 'ifgsm', 2/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
        clean_metrics, pgd_metrics_1 = test_attack(trained_model, test_loader, 'pgd', 2/255, 'None', " ", " ", "PGD", device, save_results=False)
    
    
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
                'loss_type': loss_type,
                'lambda_mean': lambda_mean,
                'beta': beta,
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
                'epoch_total_times': epoch_total_times,
                'status': 'ok'})
            
        out_dir = f'pgd_{mode}'
        os.makedirs(out_dir, exist_ok=True)

        eps_sched_label = "_"
        if mode == "baseline" and epsilon_scheduler != None:
            eps_sched_label = f"_{type_sched}_eps_sched_neprampup_{num_epochs_rampup}"
        elif mode == "baseline" and epsilon_scheduler == None:
            eps_sched_label = f"_fixed_eps_{epsilon}"
        elif mode == "ades":
            mode = f"ades_{loss_type}-lambda_mean_{lambda_mean}"
        
        with open(f'{out_dir}/grid_search_pgd_{mode}_{eps_sched_label}_lr_{lr}_{num_epochs}_alpha_{alpha_adv}_freeze.json', 'w') as f:
            json.dump(results, f, indent=4)
    
        print(f"Clean acc: {val_clean_acc:.4f} | Adv acc: {val_adv_acc:.4f}")