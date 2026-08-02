import os
import time
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import foolbox as fb
from test_attacks import test_attack

class AdaptiveMarginThreshold:
    def __init__(self, momentum=0.95, init_c=2.0):
        self.c = init_c
        self.momentum = momentum

    def update(self, margin):
        neg = -margin[margin < 0]

        if len(neg) > 0:
            batch_c = torch.quantile(neg, 0.90).item()

            self.c = (
                self.momentum*self.c
                + (1-self.momentum)*batch_c
            )

        return self.c

"""
Margin-based instance-adaptive reweighting for FGSM-AT.
 
Idea
----
Standard FGSM-AT treats every adversarial example in the batch identically when
computing the adversarial CE loss. This module adds a cheap, single-forward-pass
signal -- the post-attack logit margin -- to identify:
 
  (a) samples where the single FGSM step barely moved the prediction (still
      confidently correct)   -> weak / uninformative adversarial example
  (b) samples where FGSM just crossed the decision boundary                -> the
      "sweet spot": genuinely informative for robustness
  (c) samples where FGSM produced a confidently-flipped, large-negative-margin
      prediction             -> suspected catastrophic-overfitting driver;
                                 down-weighted
 
This is inspired by the logit-margin gating constraint used in DifAT (Ding et al.,
2025, Neurocomputing) to decide when an iterative adversarial example is "strong
enough". Here it is repurposed for the single-step FGSM setting as an instance-level
LOSS REWEIGHTING signal rather than a gate for iterative denoising -- there is no
iterative process to gate in FGSM-AT, so the mechanism is adapted rather than copied.
 
No extra backward passes, no extra attack iterations, no diffusion model: fully
compatible with an FGSM-only training constraint.
"""
# 1- logit margin computation

def compute_logit_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Binary-classification logit margin: Phi(x) = z_true(x) - z_other(x).
 
    Phi > 0  -> still classified correctly (margin = confidence of correctness)
    Phi < 0  -> misclassified (more negative = more confidently wrong)
 
    Args:
        logits: [B, 2] raw model outputs (pre-softmax) for (real, fake) or similar.
        y:      [B] integer labels in {0, 1}.
 
    Returns:
        margin: [B] tensor, Phi(x) per sample.
    """
    assert logits.dim() == 2 and logits.size(1) == 2, \
        "compute_logit_margin assumes binary classification with 2 logits per sample."
 
    y = y.view(-1, 1)
    other = 1 - y  # binary complement label
    z_true = logits.gather(1, y).squeeze(1)
    z_other = logits.gather(1, other).squeeze(1)
    return z_true - z_other

# 2- margin -> per sample weight

def margin_weighting_function(
    margin: torch.Tensor,
    c: float = 2.0,
    sigma: float = 1.5,
    weak_margin_floor: float = 0.3,
) -> torch.Tensor:
    """
    Maps the logit margin to a per-sample weight in (0, 1], down-weighting both:
      - samples with margin >> 0 (weak/uninformative adversarial examples), and
      - samples with margin << -c (suspected catastrophic-overfitting drivers).
 
    Full weight (1.0) is given to samples near the decision boundary
    (margin in a band around 0, extending a bit into negative territory up to -c).
 
    Args:
        margin: [B] tensor of Phi(x) values.
        c: threshold beyond which negative margins are considered "too confidently
           flipped" and start being down-weighted (mirrors DifAT's constraint constant).
        sigma: controls how sharply weight decays once margin < -c.
        weak_margin_floor: minimum weight given to samples with strongly positive
           margin (never fully zero them out -- they still carry some clean-adjacent
           signal via the CE loss).
 
    Returns:
        weights: [B] tensor, same device/dtype as margin.
    """
    weights = torch.ones_like(margin)
 
    # --- Down-weight "too confidently flipped" samples (margin << -c) ---
    over_flip_mask = margin < -c
    decay = torch.exp(-((margin + c) ** 2) / (2 * sigma ** 2))
    min_overflip_weight = 0.3
    decay = min_overflip_weight + (1 - min_overflip_weight) * decay
    weights = torch.where(over_flip_mask, decay, weights)
 
    # --- Down-weight "barely fooled" samples (margin >> 0) ---
    # Smoothly interpolate from weak_margin_floor (very positive margin) to 1.0
    # (margin near 0) using a sigmoid centered at 0.
    weak_mask = margin > 0
    softness = torch.sigmoid(-margin)  # -> 0 for large positive margin, -> 0.5 at margin=0
    weak_weight = weak_margin_floor + (1.0 - weak_margin_floor) * (2 * softness)
    weak_weight = torch.clamp(weak_weight, weak_margin_floor, 1.0)
    weights = torch.where(weak_mask, weak_weight, weights)
 
    return weights.detach()  # weighting must not itself receive gradient


def train_robust_with_margin(model, train_loader, val_loader, lambda_entropy, start_epoch, num_epochs,
                                  optimizer, scheduler, epsilon, eps_scheduler, criterion, device, train_losses,
                                  train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv,
                                  seed, entropy_flag=False, has_eps_sched = False, adaptive=False, save_model=True,
                                  # >>> MARGIN: new args, all default to previous behaviour when margin_weighting=False
                                  margin_weighting=False, margin_c=2.0, margin_sigma=1.5, margin_weak_floor=0.3):
    train_loss = 0.0
    history = {}
    val_losses = []
 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    
    val_acc_adv = 0
    clean_acc   = 0

        # >>> TIMING: per-epoch and cumulative wall-clock time, in seconds.
    # Useful for your FGSM-AT vs PGD-AT speed comparison in the thesis.
    epoch_train_times = []
    epoch_val_times = []
    epoch_total_times = []
    training_start_time = time.perf_counter()
 
    # >>> MARGIN: per-epoch diagnostics, useful for correlating with catastrophic
    # overfitting onset in your existing plots
    margin_diagnostics = {"mean_margin": [], "mean_weight": [],
                          "frac_overflip_downweighted": [], "frac_weak_downweighted": []}
 
    if has_eps_sched == False:
        print(f"Training with {epsilon} epsilon")
    else:
        print(f"Training with epsilon scheduler type {eps_scheduler.type}")
        
    if margin_weighting:
        print(f"Margin-based adversarial loss reweighting: ENABLED (c={margin_c}, sigma={margin_sigma})")
 
    for epoch in range(start_epoch, num_epochs):

        # >>> TIMING: mark epoch start
        epoch_start_time = time.perf_counter()

        #TRAINING
        model.train()
        train_loss = 0.0
 
        if has_eps_scheduler == True:

            if eps_scheduler.should_stop():
                break
            else:
                current_eps = eps_scheduler.get_epsilon(val_acc_adv, clean_acc, epoch)

        else:
            current_eps = epsilon
        #adaptive_threshold = AdaptiveMarginThreshold()
 
        # I block update statistics of BatchNorm
        freeze_bn(model)
        train_metrics_clean.reset_epoch()
        train_metrics_adv.reset_epoch()
        val_metrics_clean.reset_epoch()
        val_metrics_adv.reset_epoch()
        print(f"Epoch {epoch+1} | eps: {current_eps:.4f}")
 
        # >>> MARGIN: per-epoch accumulators for diagnostics
        epoch_margins = []
        epoch_weights = []
 
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            if batch is None:
                continue
            imgs_raw, y, _ = batch
            imgs_raw, y = imgs_raw.to(device), y.to(device).long().view(-1)
 
            # Clean Forward pass
            optimizer.zero_grad()
            imgs_raw.requires_grad = True
            imgs = normalize(imgs_raw)
            logits_clean = model(imgs)
 
            # Guard: skip batch if NaNs detected
            if torch.isnan(logits_clean).any() or torch.isinf(logits_clean).any():
                print(f"[WARNING] NaN/Inf in logits, skipping")
                continue
 
            loss_clean = criterion(logits_clean, y)
 
            # RS-FGSM: random start before the FGSM step, to mitigate
            # catastrophic overfitting (Wong et al. / RS-FGSM).
            # 1) sample a uniform random perturbation within the eps-ball
            # 2) clamp so imgs_raw + delta stays a valid image in [0, 1]
            # 3) take the FGSM step from THIS randomly-perturbed starting point
            delta = torch.empty_like(imgs_raw).uniform_(-current_eps, current_eps)
            delta = torch.clamp(imgs_raw.detach() + delta, 0, 1) - imgs_raw.detach()
            imgs_start = (imgs_raw.detach() + delta).clone().requires_grad_(True)
 
            imgs_start_norm = normalize(imgs_start)
            logits_start = model(imgs_start_norm)
            loss_start = criterion(logits_start, y)
 
            # I compute the gradient respect to the (randomly-started) image
            grad_imgs = torch.autograd.grad(loss_start, imgs_start, retain_graph=False, create_graph=False)[0]
            imgs_adv = imgs_start.detach() + current_eps * grad_imgs.sign()
            imgs_adv = torch.clamp(imgs_adv, 0, 1)
            imgs_adv = normalize(imgs_adv.detach())
 
            #Adversarial forward pass
            logits_adv = model(imgs_adv)
 
            if torch.isnan(logits_adv).any() or torch.isinf(logits_adv).any():
                print(f"[WARNING] NaN/Inf in adversarial logits, skipping")
                continue

            # >>> MARGIN: compute per-sample margin + weight, then weighted adv loss
            if margin_weighting:
                with torch.no_grad():
                    margin = compute_logit_margin(logits_adv, y)
                    # adapt c, it is the 10th percentile of the absolute margins
                    #c = adaptive_threshold.update(margin)
                    weights = margin_weighting_function(
                        margin, c=margin_c, sigma=margin_sigma, weak_margin_floor=margin_weak_floor
                    )
                per_sample_loss_adv = F.cross_entropy(logits_adv, y, reduction="none")
                #loss_adv = (weights * per_sample_loss_adv).mean()
                loss_adv = (weights * per_sample_loss_adv).sum() / weights.sum()

                #print(weights.min())
                #print(weights.max())
                #print(weights.mean())
                #print(torch.quantile(
                #    weights,
                #    torch.tensor([0,0.25,0.5,0.75,1], device=weights.device)
                #))
#
                #print("adv CE      :", per_sample_loss_adv.mean().item())
                #print("weighted CE :", loss_adv.item())
                #print("clean CE    :", loss_clean.item())
#
                #corr = torch.corrcoef(torch.stack([
                #    per_sample_loss_adv.detach(),
                #    weights.detach()
                #]))[0,1]
#
                #print(corr)
#
                #print(torch.quantile(margin, torch.tensor([0.0,0.25,0.5,0.75,1.0], device=margin.device)))
 
                # accumulate diagnostics
                epoch_margins.append(margin.detach())
                epoch_weights.append(weights.detach())
            else:
                loss_adv = criterion(logits_adv, y)
 
            if entropy_flag == True:
                # compute entropy
                entropy_clean = entropy_penalty(logits_clean)
                entropy_adv = entropy_penalty(logits_adv)
                # adding entropy penalty to the loss
                loss = 0.5 * (loss_clean + loss_adv) - lambda_entropy * entropy_adv
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
 
        # >>> MARGIN: aggregate this epoch's diagnostics and log them
        if margin_weighting and len(epoch_margins) > 0:
            all_margins = torch.cat(epoch_margins)
            all_weights = torch.cat(epoch_weights)
            print(all_weights.mean())
            margin_diagnostics["mean_margin"].append(all_margins.mean().item())
            margin_diagnostics["mean_weight"].append(all_weights.mean().item())
            margin_diagnostics["frac_overflip_downweighted"].append(
                (all_margins < -margin_c).float().mean().item()
            )
            margin_diagnostics["frac_weak_downweighted"].append(
                (all_margins > 0).float().mean().item()
            )
            print(f"[MARGIN] mean_margin={margin_diagnostics['mean_margin'][-1]:.4f} "
                  f"mean_weight={margin_diagnostics['mean_weight'][-1]:.4f} "
                  f"frac_overflip={margin_diagnostics['frac_overflip_downweighted'][-1]:.4f} "
                  f"frac_weak={margin_diagnostics['frac_weak_downweighted'][-1]:.4f}")

            print(f"average adv loss: {loss_adv.mean()}")
            print(f"clean loss:{loss_clean}")
            
        # >>> TIMING: sync CUDA before reading the clock, so GPU work actually
        # finishes before we stop the timer (otherwise async kernel launches
        # make this measurement meaningless)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_train_time = time.perf_counter() - epoch_start_time
        epoch_train_times.append(epoch_train_time)
 
        val_start_time = time.perf_counter()
 
        #VALIDATION
        model.eval()
        val_metrics_clean.reset_epoch()
        val_metrics_adv.reset_epoch()
        val_loss = 0.0
 
        pbar = tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
            if batch is None:
                continue
            imgs_raw, y, _ = batch
            imgs_raw, y = imgs_raw.to(device), y.to(device).long().view(-1)

            # no reweighting applied here, since reweighting is a TRAINING-time
            # mechanism for shaping gradients, not an evaluation metric)
            # Adversarial validation (unchanged in spirit -- validation stays on plain
            # FGSM loss, no reweighting applied here, since reweighting is a
            # TRAINING-time mechanism for shaping gradients, not an evaluation metric.
            # RS-FGSM's random start IS applied here too, so validation attack matches
            # the attack the model was actually trained against.)
            with torch.enable_grad():
                imgs_raw = imgs_raw.detach()
                delta = torch.empty_like(imgs_raw).uniform_(-current_eps, current_eps)
                delta = torch.clamp(imgs_raw + delta, 0, 1) - imgs_raw
                imgs_start = (imgs_raw + delta).clone().requires_grad_(True)
 
                imgs_norm = normalize(imgs_start)
                logits_clean = model(imgs_norm)
                loss_tmp = criterion(logits_clean, y)
                grad_imgs = torch.autograd.grad(loss_tmp, imgs_start)[0]
 
            imgs_adv = imgs_start.detach() + current_eps * grad_imgs.sign()
            imgs_adv = torch.clamp(imgs_adv, 0, 1)
 
            with torch.no_grad():
                logits_clean = model(normalize(imgs_raw.detach()))
                loss_clean = criterion(logits_clean, y)
                logits_adv = model(normalize(imgs_adv.detach()))
                loss_adv = criterion(logits_adv, y)
                if entropy_flag == True:
                    entropy_clean = entropy_penalty(logits_clean)
                    entropy_adv = entropy_penalty(logits_adv)
                    loss = 0.5 * (loss_clean + loss_adv) - lambda_entropy * entropy_adv
                else:
                    loss = 0.5 * loss_clean + 0.5 * loss_adv
 
                val_loss += loss.item() * imgs_norm.size(0)
                epoch_val_loss = val_loss / len(val_loader.dataset)
 
                probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_clean.update(y, probs_clean)
                probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
                val_metrics_adv.update(y, probs_adv)
 
        val_results_clean = val_metrics_clean.compute()
        val_results_adv = val_metrics_adv.compute()
        val_losses.append(epoch_val_loss)
 
        val_acc_adv = val_metrics_adv.accuracy_list[epoch]
        clean_acc = val_metrics_clean.accuracy_list[epoch]

        val_acc_adv = val_metrics_adv.accuracy_list[epoch]
        clean_acc = val_metrics_clean.accuracy_list[epoch] 

        # >>> TIMING: stop validation timer, compute epoch total, log it
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_val_time = time.perf_counter() - val_start_time
        epoch_val_times.append(epoch_val_time)
 
        epoch_total_time = epoch_train_time + epoch_val_time
        epoch_total_times.append(epoch_total_time)
 
        print(f"[TIMING] Epoch {epoch+1}: train={epoch_train_time:.2f}s | "
              f"val={epoch_val_time:.2f}s | total={epoch_total_time:.2f}s | "
              f"cumulative={time.perf_counter() - training_start_time:.2f}s")
 
        # >>> TIMING: total wall-clock time for the whole call (all epochs run)
        total_training_time = time.perf_counter() - training_start_time
        print(f"[TIMING] Total training time: {total_training_time:.2f}s "
              f"({total_training_time/60:.2f} min) over {len(epoch_total_times)} epochs "
              f"(avg {sum(epoch_total_times)/max(len(epoch_total_times),1):.2f}s/epoch)")
        
        avg_training_time = sum(epoch_total_times) / max(len(epoch_total_times), 1)

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

    if has_eps_scheduler == True:
        epsilon_tag = f"_epsilonsched_{eps_scheduler.type}"
    else:
        epsilon_tag = f"_epsilon_{epsilon}"
 
    eps_sched_folder = ""
    save_path = ""
    history_path = ""
 
    folder_name = "with_eps_scheduler"
    # >>> MARGIN: tag filenames so margin-weighted runs don't overwrite baseline runs
    margin_tag = "_marginweighted_2" if margin_weighting else ""
    save_path =  f'{MODELS_DIR}/{folder_name}/resnet50_{attack_name}_epoch_{epoch}_seed_{seed}{epsilon_tag}{margin_tag}.pt'
    history_path = f"history/{folder_name}/history_{attack_name}_epoch_{epoch}_seed_{seed}{epsilon_tag}{margin_tag}.json"
 
    print(f"Saving models in {history_path}")
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
        "val_f1_adv": val_metrics_adv.f1_list,
        "val_precision_adv": val_metrics_adv.precision_list,
        "val_recall_adv": val_metrics_adv.recall_list,
        "val_accuracy_adv": val_metrics_adv.accuracy_list,
        "train_asr": train_metrics_adv.asr_list,
        "train_epsilon": epsilon,
        # >>> MARGIN: persist diagnostics for later plotting (e.g. in plot_heatmaps.py)
        "margin_weighting_enabled": margin_weighting,
        "margin_diagnostics": margin_diagnostics,
        # >>> TIMING: persist per-epoch and total wall-clock times (seconds),
        # for the FGSM-AT vs PGD-AT speed comparison in the thesis
        "epoch_train_times_sec": epoch_train_times,
        "epoch_val_times_sec": epoch_val_times,
        "epoch_total_times_sec": epoch_total_times,
        "total_training_time_sec": total_training_time,
        "avg_epoch_time_sec": sum(epoch_total_times) / max(len(epoch_total_times), 1)
    }

    save_history_json(history, history_path)

    if save_model == True:
        print(f"Saving models in {save_path}")
 
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
        }, save_path)

 
    return model, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses, epoch_train_times, epoch_val_times, epoch_val_times, epoch_total_times, total_training_time

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

    # Generate all combinations
    #keys   = list(param_grid.keys())
    #values = list(param_grid.values())

    epsilons = [2/255]

    # TRAINING WITH FGSM AUX LOSS
    for epsilon in epsilons:
    
        torch.cuda.empty_cache()
    
        results     = []
    
        print("Initializing Data loaders ......")
        train_loader, val_loader, test_loader = get_data_loaders(transform, params['batch_size'])
    
        checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
    
        print(f"Start training PGD-AT")
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
    
        # PARAMETERS #################
        num_epochs = 15
        alpha_adv = 0.5
        entropy_flag = False
        has_eps_scheduler = True
        has_gradient_penalty = False
        has_fgsm_aux_loss = False
        margin_weighting = False
        lambda_fgsm = 0
    
        # EPSILON SCHEDULER
        type = 'linear'
        eps_scheduler = CurriculumEpsilonScheduler(
                eps_start=0/255, eps_end=8/255,
                num_epochs_rampup=num_epochs, type=type,
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

        margin_tag = "_"
        if margin_weighting == True:
            margin_tag = "_marginweighted_2"
    
        
        # FGSM-AT with MARGIN LOGITS
        
        trained_model, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses, epoch_train_times, epoch_val_times, epoch_val_times, epoch_total_times, total_training_time = train_robust_with_margin(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        lambda_entropy=params["lambda_entropy"],
        start_epoch=start_epoch, 
        num_epochs=num_epochs, 
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
        epsilon=epsilon,
        eps_scheduler=eps_scheduler,
        entropy_flag=entropy_flag,
        has_eps_sched=has_eps_scheduler,
        save_model=True,
        margin_weighting=margin_weighting)
    
        val_adv_acc = val_metrics_adv.accuracy_list[num_epochs-1]
        val_clean_acc = val_metrics_clean.accuracy_list[num_epochs-1]
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
         # testing
        clean_metrics, rs_fgsm_metrics_1 = test_attack(trained_model, test_loader, 'rs_fgsm', 2/255, 'None', " ", " ", "FGSM (eps=2/255)", device, save_results=False)
        clean_metrics, rs_fgsm_metrics_2 = test_attack(trained_model, test_loader, 'rs_fgsm', 8/255, 'None', " ", " ", "FGSM (eps=8/255)", device, save_results=False)
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
                'test_rsfgsm_small_acc': rs_fgsm_metrics_1.accuracy_list[0],
                'test_rsfgsm_small_auc': rs_fgsm_metrics_1.auc_list[0],
                'test_rsfgsm_small_asr': rs_fgsm_metrics_1.asr_list[0],
                'test_rsfgsm_big_acc': rs_fgsm_metrics_2.accuracy_list[0],
                'test_rsfgsm_big_auc': rs_fgsm_metrics_2.auc_list[0],
                'test_rsfgsm_big_asr': rs_fgsm_metrics_2.asr_list[0],
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
                'margin': False,
                'status': 'ok',
                "epoch_train_times_sec": epoch_train_times,
                "epoch_val_times_sec": epoch_val_times,
                "epoch_total_times_sec": epoch_total_times,
                "total_training_time_sec": total_training_time,
                "avg_epoch_time_sec": sum(epoch_total_times) / max(len(epoch_total_times), 1)
                })
        
        with open(f'results_grid_search/linear_per_epoch/grid_search_{attack}_{grad_penalty}_{eps_label}_with_test_results_Cyclic_{num_epochs}_alpha_{alpha_adv}_{fgsm_aux_loss}{margin_tag}.json', 'w') as f:
            json.dump(results, f, indent=4)
    
        print(f"Clean acc: {val_clean_acc:.4f} | Adv acc: {val_adv_acc:.4f}")

