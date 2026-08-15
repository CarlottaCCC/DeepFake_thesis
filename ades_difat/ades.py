"""
ADES: Adaptive Dynamic Epsilon Scheduling, applied to PGD-AT.
 
Reference: "Learnable Dynamic Epsilon Scheduling for Instance-Aware Adversarial
Training" (arXiv:2506.12733). Original paper operates on CIFAR-10/100 with
WideResNet; this module adapts the mechanism to a binary deepfake detector
(ResNet50) with PGD as the inner attack.
 
Core idea: instead of one fixed epsilon for every sample, a small learnable MLP
(the "scheduler") fuses three per-sample signals -- gradient norm, prediction
entropy, and MC-dropout uncertainty -- into a scalar in [0, 1] that sets the
sample's own perturbation budget: eps_x = eps_min + lambda * scheduler(signals).
 
IMPORTANT ARCHITECTURE REQUIREMENT
-----------------------------------
MC-dropout uncertainty requires at least one nn.Dropout layer in the model
(commonly inserted right before the final classification layer). If your
ResNet50 doesn't already have one, add e.g.:
    model.fc = nn.Sequential(nn.Dropout(p=0.3), model.fc)
before using this module. Without it, `estimate_mc_uncertainty` will return
all-zero uncertainty (dropout has no effect), which silently degrades ADES to
a 2-signal (gradient norm + entropy) scheduler -- still usable, just weaker
than the paper's 3-signal version.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
os.environ['MPLCONFIGDIR'] = "/work/project"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from utils import *

 
 
# ---------------------------------------------------------------------------
# 1. Signal extraction
# ---------------------------------------------------------------------------
 
def gradient_norm_signal(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                          criterion, normalize) -> torch.Tensor:
    """Per-sample L2 norm of the input gradient (local loss sensitivity)."""
    x_ = x.clone().detach().requires_grad_(True)
    logits = model(normalize(x_))
    loss = criterion(logits, y)
    grad = torch.autograd.grad(loss, x_, create_graph=False)[0]
    return grad.view(grad.size(0), -1).norm(2, dim=1)
 
 
def confidence_entropy_signal(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample prediction entropy (ambiguity signal)."""
    probs = F.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
 
 
def enable_mc_dropout(model: nn.Module, mc_dropout_p: float = None):
    """
    Puts ONLY nn.Dropout modules into train mode, leaving BatchNorm and
    everything else in eval mode. Call model.eval() first, then this.
 
    Args:
        mc_dropout_p: if given, temporarily OVERRIDES each Dropout module's `p`
            for the duration of MC sampling (see restore_dropout_p). Use a
            higher value than your model's normal training dropout rate if
            that rate is too low to produce meaningfully different predictions
            across the T stochastic passes (a common issue: a base model
            tuned with e.g. p=0.1-0.2 for standard regularization often gives
            near-zero MC variance, i.e. an uninformative uncertainty signal).
    """
    model.eval()
    original_ps = {}
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            if mc_dropout_p is not None:
                original_ps[m] = m.p
                m.p = mc_dropout_p
    return original_ps  # empty dict if mc_dropout_p was None
 
 
def restore_dropout_p(original_ps: dict):
    """Restores each Dropout module's original `p` after MC sampling."""
    for m, p in original_ps.items():
        m.p = p
 
 
@torch.no_grad()
def estimate_mc_uncertainty(model: nn.Module, x_raw: torch.Tensor, normalize, T: int = 3,
                             mc_dropout_p: float = None) -> torch.Tensor:
    """
    Per-sample epistemic uncertainty via T stochastic forward passes with
    dropout active. Returns mean predictive variance across classes.
    Requires the model to contain at least one nn.Dropout layer (see module
    docstring) -- otherwise this returns all-zero tensors.
 
    Args:
        mc_dropout_p: optional override, see enable_mc_dropout. If your base
            model's normal dropout rate is too low to produce informative MC
            variance, pass a higher value here (e.g. 0.3-0.5) -- it only
            applies during these T sampling passes and is restored immediately
            after, so it never affects the actual training forward/backward
            passes elsewhere in the loop.
    """
    original_ps = enable_mc_dropout(model, mc_dropout_p=mc_dropout_p)
    x_norm = normalize(x_raw)
    probs_samples = []
    for _ in range(T):
        logits = model(x_norm)
        probs_samples.append(F.softmax(logits, dim=1))
    restore_dropout_p(original_ps)
    model.eval()  # restore full eval mode afterwards
 
    probs_stack = torch.stack(probs_samples, dim=0)  # [T, B, num_classes]
    variance = probs_stack.var(dim=0)  # [B, num_classes]
    return variance.mean(dim=1)  # [B]
 
 
def normalize_signal(sig: torch.Tensor) -> torch.Tensor:
    """Batch-wise min-max normalization to [0, 1], matching the ADES paper."""
    lo, hi = sig.min(), sig.max()
    if (hi - lo).item() < 1e-8:
        return torch.zeros_like(sig)
    return (sig - lo) / (hi - lo)
 
 
# ---------------------------------------------------------------------------
# 2. Learnable fusion scheduler
# ---------------------------------------------------------------------------
 
class LearnableEpsilonScheduler(nn.Module):
    """
    Lightweight 2-layer MLP fusing normalized [grad_norm, entropy, uncertainty]
    into a scalar sigma(x) in [0, 1], jointly trained end-to-end with the model.
    """
 
    def __init__(self, hidden_dim: int = 16, num_signals: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_signals, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        #nn.init.constant_(self.net[2].bias, -3.0)
 
    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        # signals: [B, num_signals] already normalized to [0, 1]
        return self.net(signals).squeeze(1)  # [B]
 
 
def compute_adaptive_epsilon(scheduler: EpsilonScheduler, model: nn.Module,
                              x_raw: torch.Tensor, y: torch.Tensor, criterion, normalize,
                              eps_min: float, eps_lambda: float, mc_passes: int = 3,
                              mc_dropout_p: float = 0.3):
    """
    Runs the full ADES pipeline for one batch: extract signals -> normalize ->
    fuse via scheduler -> map to per-sample epsilon.
 
    Args:
        mc_dropout_p: optional dropout-rate override used ONLY during MC
            uncertainty sampling (see estimate_mc_uncertainty). Leave as None
            to use the model's own dropout layer(s) at whatever rate they're
            currently set to (e.g. the base model's pretrained/tuned rate).
 
    Returns:
        eps_x: [B] tensor of per-sample perturbation budgets
        scheduler_output: [B] tensor, sigma(x) (kept for backprop into scheduler)
    """
    g = gradient_norm_signal(model, x_raw, y, criterion, normalize)
    with torch.no_grad():
        logits_clean = model(normalize(x_raw))
    h = confidence_entropy_signal(logits_clean)
    u = estimate_mc_uncertainty(model, x_raw, normalize, T=mc_passes, mc_dropout_p=mc_dropout_p)
 
    g_n, h_n, u_n = normalize_signal(g), normalize_signal(h), normalize_signal(u)
    signals = torch.stack([g_n, h_n, u_n], dim=1)  # [B, 3]
 
    sigma = scheduler(signals)  # [B], differentiable w.r.t. scheduler params
    #print("sigma requires_grad:", sigma.requires_grad)
    #print("sigma grad_fn:", sigma.grad_fn)
    sigma.retain_grad()

    #heuristic 1
    eps_x = eps_min + eps_lambda * sigma

    #print(g.requires_grad, g.grad_fn)
    #print(h.requires_grad, h.grad_fn)
    #print(u.requires_grad, u.grad_fn)
    #print(signals.requires_grad, signals.grad_fn)
    #print(sigma.requires_grad, sigma.grad_fn)

    return eps_x, sigma
 
 
# ---------------------------------------------------------------------------
# 3. PGD attack with per-sample epsilon
# ---------------------------------------------------------------------------
 
def pgd_attack_adaptive_eps(model: nn.Module, normalize, x: torch.Tensor, y: torch.Tensor,
                             eps_x: torch.Tensor, alpha: float, steps: int,
                             clamp_min: float = 0.0, clamp_max: float = 1.0) -> torch.Tensor:
    """
    Standard PGD, but the L_inf ball radius is per-sample (eps_x: [B]) instead
    of a single global epsilon.
    """
    eps_x_ = eps_x.view(-1, 1, 1, 1)
    delta = torch.empty_like(x).uniform_(-1, 1) * eps_x_
    delta = torch.clamp(x + delta, clamp_min, clamp_max) - x
    delta = delta.detach().requires_grad_(True)
 
    for _ in range(steps):
        logits = model(normalize(x + delta))
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = delta.detach() + alpha * grad.sign()
        delta = torch.max(torch.min(delta, eps_x_), -eps_x_)  # per-sample clip
        delta = torch.clamp(x + delta, clamp_min, clamp_max) - x
        delta.requires_grad_(True)

    #print(f"Scheduler loss:{loss}")
 
    return torch.clamp(x + delta.detach(), clamp_min, clamp_max)

def pgd_attack_adaptive_eps_differentiable(model: nn.Module, normalize, x: torch.Tensor, y: torch.Tensor,
                             eps_x: torch.Tensor, alpha: float, steps: int,
                             clamp_min: float = 0.0, clamp_max: float = 1.0) -> torch.Tensor:
    """
    Standard PGD, but the L_inf ball radius is per-sample (eps_x: [B]) instead
    of a single global epsilon.
    """
    eps_x_ = eps_x.view(-1, 1, 1, 1)
    delta = torch.empty_like(x).uniform_(-1, 1) * eps_x_
    delta = torch.clamp(x + delta, clamp_min, clamp_max) - x
    delta = delta.requires_grad_(True) # no detach()
 
    for _ in range(steps-1):
        logits = model(normalize(x + delta))
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = delta.detach() + alpha * grad.sign()
        delta = torch.max(torch.min(delta, eps_x_), -eps_x_)  # per-sample clip
        delta = torch.clamp(x + delta, clamp_min, clamp_max) - x
        delta.requires_grad_(True)

    # I differentiate just at the last step
    logits = model(normalize(x + delta))
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, delta, create_graph=True)[0] # I keep the computational graph
    delta = delta + alpha * grad.sign() # no detach
    # check
    #active = (delta.abs() >= eps_x_).float().mean()
    #print(f"active: {active}")
    delta = torch.max(torch.min(delta, eps_x_), -eps_x_)  # per-sample clip
    #print(f"delta.requires_grad:{delta.requires_grad}")
    #print(f"delta.grad_fn:{delta.grad_fn}")
    delta = torch.clamp(x + delta, clamp_min, clamp_max) - x
    delta.requires_grad_(True)

    #print(f"Scheduler loss:{loss}")
    #print(delta.grad_fn)

    #img_adv = torch.clamp(x + delta.detach(), clamp_min, clamp_max)
    #print(img_adv.min().item(), img_adv.max().item(), img_adv.mean().item(), img_adv.std().item())
 
    return torch.clamp(x + delta, clamp_min, clamp_max)

def check_mc_dropout_calibration(model, val_loader, device, T=10,
                                  candidate_ps=(None, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """
    Runs estimate_mc_uncertainty on a single validation batch for each
    candidate dropout probability and prints the resulting mean/std variance,
    so you can pick the smallest mc_dropout_p that still gives a clearly
    non-zero, informative spread.
 
    Args:
        model: your classifier (must contain >=1 nn.Dropout layer).
        val_loader: any DataLoader yielding (imgs, y, ...) batches.
        device: torch device.
        T: number of stochastic MC passes per check (higher = more reliable
           variance estimate, at the cost of T forward passes per candidate).
        candidate_ps: values to sweep. `None` means "use the model's own
           current dropout rate, unmodified".
    """
    model.eval()
 
    # grab exactly one batch
    batch = next(iter(val_loader))
    imgs_raw, y, _ = batch
    imgs_raw = imgs_raw.to(device)
 
    print(f"{'mc_dropout_p':>14} | {'mean variance':>14} | {'std variance':>13} | {'max variance':>13}")
    print("-" * 62)
 
    for p in candidate_ps:
        variance = estimate_mc_uncertainty(model, imgs_raw, normalize, T=T, mc_dropout_p=p)
        label = "model default" if p is None else f"{p:.2f}"
        print(f"{label:>14} | {variance.mean().item():>14.6f} | "
              f"{variance.std().item():>13.6f} | {variance.max().item():>13.6f}")
 
    print("\nInterpretation:")
    print("  - mean variance near 0.0 (e.g. < 1e-4) => uninformative, bump mc_dropout_p up")
    print("  - mean variance that grows steadily as p increases => dropout IS reaching the")
    print("    classifier head correctly; pick the smallest p where variance is clearly")
    print("    non-zero and samples are distinguishable (check std/max too, not just mean --")
    print("    you want spread ACROSS samples, not just a uniformly large number)")
    print("  - if variance stays ~0 even at p=0.5 => check that your model actually contains")
    print("    an nn.Dropout layer, and that it sits before the final classification layer")
    print("    (a Dropout layer whose output feeds into something with no further learnable")
    print("    weights won't affect the logits at all).")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    set_seed(42)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])

    train_loader, val_loader, test_loader = get_data_loaders(transform, 16)
    
    checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
    
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    check_mc_dropout_calibration(model, val_loader, device)
