"""
DifAT: Diffusion Adversarial Training, applied to PGD-AT for deepfake detection.
 
Reference: "Enhancing robust generalization through appropriate adversarial
example attack intensity" (Ding et al., Neurocomputing 2025). Original paper
uses CIFAR-10/100/Tiny-ImageNet with a DDPM trained on those datasets.

"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# ---------------------------------------------------------------------------
# 1. Diffusion purification wrapper (S=1, per the paper's own best setting)
# ---------------------------------------------------------------------------
 
class DiffusionPurifier:
 
    def __init__(self, unet: nn.Module, scheduler, device, diffusion_step: int = 1):
        self.unet = unet.to(device).eval()
        self.scheduler = scheduler
        self.device = device
        self.s = diffusion_step  # S=1 per the paper's ablation (Fig. 4/5)
 
    @torch.no_grad()
    def purify(self, x_01: torch.Tensor) -> torch.Tensor:

        x = x_01 * 2.0 - 1.0  # [0,1] -> [-1,1] for the diffusion model
 
        t = torch.full((x.size(0),), self.s, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_noisy = self.scheduler.add_noise(x, noise, t)  # forward diffusion, S steps
 
        # single reverse step: predict noise, remove it
        pred_noise = self.unet(x_noisy, t).sample
        step_out = self.scheduler.step(pred_noise, self.s, x_noisy)
        x_denoised = step_out.prev_sample
 
        x_denoised = (x_denoised + 1.0) / 2.0  # back to [0,1]
        return torch.clamp(x_denoised, 0.0, 1.0)
 
 
# ---------------------------------------------------------------------------
# 2. Logit margin + attack-strength constraint (Eq. 7-8 in the paper)
# ---------------------------------------------------------------------------
 
def compute_logit_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    
    y_idx = y.view(-1, 1)
    other_idx = 1 - y_idx
    z_true = logits.gather(1, y_idx).squeeze(1)
    z_other = logits.gather(1, other_idx).squeeze(1)
    return z_true - z_other
 
 
# ---------------------------------------------------------------------------
# 3. DPGD attack (Algorithm 1 in the paper)
# ---------------------------------------------------------------------------
 
def dpgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
                 alpha: float, steps: int, purifier: DiffusionPurifier, normalize,
                 margin_c: float = 0.0, control_factor_tau: int = 1,
                 clamp_min: float = 0.0, clamp_max: float = 1.0) -> torch.Tensor:

    x0 = x.clone().detach()
    delta = torch.zeros_like(x)
    con = torch.zeros(x.size(0), device=x.device)
 
    for t in range(steps):
        x_t = torch.clamp(x0 + delta, clamp_min, clamp_max).detach().requires_grad_(True)
        logits = model(normalize(x_t))
 
        with torch.no_grad():
            margin = compute_logit_margin(logits, y)
            satisfied_now = (-margin) >= margin_c
            con = torch.where(satisfied_now, con + 1, con)
            should_purify = con >= control_factor_tau
 
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_t)[0]
 
        delta = delta.detach() + alpha * grad.sign()
        delta = torch.clamp(delta, -eps, eps)
        x_next = torch.clamp(x0 + delta, clamp_min, clamp_max)
 
        # >>> purify only the samples in the batch that satisfy the constraint
        if should_purify.any():
            purified = purifier.purify(x_next)
            mask = should_purify.view(-1, 1, 1, 1).float()
            x_next = mask * purified + (1 - mask) * x_next
            delta = (x_next - x0).detach()
 
    return torch.clamp(x0 + delta, clamp_min, clamp_max).detach()


def dpgd_ades_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps_x: torch.Tensor,
                 alpha: float, steps: int, purifier: DiffusionPurifier, normalize,
                 margin_c: float = 0.0, control_factor_tau: int = 1,
                 clamp_min: float = 0.0, clamp_max: float = 1.0) -> torch.Tensor:

    x0 = x.clone().detach()
    eps_x_ = eps_x.view(-1, 1, 1, 1)
    delta = torch.zeros_like(x)
    con = torch.zeros(x.size(0), device=x.device)
 
    for t in range(steps):
        x_t = torch.clamp(x0 + delta, clamp_min, clamp_max).detach().requires_grad_(True)
        logits = model(normalize(x_t))
 
        with torch.no_grad():
            margin = compute_logit_margin(logits, y)
            satisfied_now = (-margin) >= margin_c
            con = torch.where(satisfied_now, con + 1, con)
            should_purify = con >= control_factor_tau
 
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_t)[0]
 
        delta = delta.detach() + alpha * grad.sign()
        delta = torch.max(torch.min(delta, eps_x_), -eps_x_)  # per-sample clip like
        #delta = torch.clamp(delta, -eps, eps)
        x_next = torch.clamp(x0 + delta, clamp_min, clamp_max)
 
        # >>> purify only the samples in the batch that satisfy the constraint
        if should_purify.any():
            purified = purifier.purify(x_next)
            mask = should_purify.view(-1, 1, 1, 1).float()
            x_next = mask * purified + (1 - mask) * x_next
            delta = (x_next - x0).detach()
 
    return torch.clamp(x0 + delta, clamp_min, clamp_max).detach()
 
 
# ---------------------------------------------------------------------------
# 4. DifAT loss (Eq. 11): min over relatively-weak DPGD examples, not the
#    strongest possible ones
# ---------------------------------------------------------------------------
 
def difat_loss(model: nn.Module, x_adv: torch.Tensor, y: torch.Tensor,
                criterion) -> torch.Tensor:

    logits_adv = model(x_adv)
    return criterion(logits_adv, y), logits_adv
 
