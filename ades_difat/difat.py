"""
DifAT: Diffusion Adversarial Training, applied to PGD-AT for deepfake detection.
 
Reference: "Enhancing robust generalization through appropriate adversarial
example attack intensity" (Ding et al., Neurocomputing 2025). Original paper
uses CIFAR-10/100/Tiny-ImageNet with a DDPM trained on those datasets.
 
Core idea: standard PGD keeps attacking for all T iterations regardless of
whether the adversarial example already fools the model. This lets adversarial
examples drift far from the clean data distribution, causing the model to
overfit to a distorted "adversarial distribution" and lose clean accuracy.
DPGD fixes this by checking, at each iteration, whether the example already
satisfies a minimum attack-strength constraint (logit margin below a
threshold c). Once it does (tracked via a counter `con` vs. threshold `tau`),
the example is pulled back toward the clean distribution via one diffuse+
denoise step, before continuing to attack -- keeping it effective but closer
to the original image than unconstrained PGD would produce.
 
PRETRAINED DIFFUSION MODEL
---------------------------
Per the paper's own ablation, S=1 (a single noise step / single denoise step)
gives the best robustness-generalization tradeoff -- more steps over-purify
and destroy the adversarial signal. This means a pretrained face-diffusion
checkpoint (e.g. a DDPM trained on CelebA-HQ/FFHQ via HuggingFace `diffusers`)
is a reasonable starting point rather than training one from scratch on FF++,
since the purification only needs to be locally accurate for one small step,
not generate faces from scratch. Swap `DiffusionPurifier`'s internals for a
from-scratch-trained model later if the pretrained domain gap turns out to
hurt purification quality in practice.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# ---------------------------------------------------------------------------
# 1. Diffusion purification wrapper (S=1, per the paper's own best setting)
# ---------------------------------------------------------------------------
 
class DiffusionPurifier:
    """
    Thin wrapper around a pretrained DDPM/DDIM (e.g. from HuggingFace
    `diffusers`) that performs ONE forward diffusion step followed by ONE
    reverse (denoising) step, matching the paper's S=1 finding.
 
    Expects a `diffusers`-style scheduler + UNet pair. Example setup:
 
        from diffusers import DDPMPipeline, DDPMScheduler
        pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
        purifier = DiffusionPurifier(pipe.unet, pipe.scheduler, device=device)
 
    NOTE: input images to this wrapper must be in [-1, 1] (standard diffusion
    convention), NOT the [0, 1] range used elsewhere in your pipeline, and
    NOT ImageNet-normalized. Convert at the call site (see `purify` docstring).
    """
 
    def __init__(self, unet: nn.Module, scheduler, device, diffusion_step: int = 1):
        self.unet = unet.to(device).eval()
        self.scheduler = scheduler
        self.device = device
        self.s = diffusion_step  # S=1 per the paper's ablation (Fig. 4/5)
 
    @torch.no_grad()
    def purify(self, x_01: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_01: [B, C, H, W] image batch in [0, 1] (your pipeline's convention).
        Returns:
            purified image batch in [0, 1], same shape.
        """
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
    """Binary logit margin Phi(x) = z_true(x) - z_other(x)."""
    y_idx = y.view(-1, 1)
    other_idx = 1 - y_idx
    z_true = logits.gather(1, y_idx).squeeze(1)
    z_other = logits.gather(1, other_idx).squeeze(1)
    return z_true - z_other
 
 
# ---------------------------------------------------------------------------
# 3. DPGD attack (Algorithm 1 in the paper)
# ---------------------------------------------------------------------------
 
def dpgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
                 alpha: float, steps: int, purifier: DiffusionPurifier,
                 margin_c: float = 0.0, control_factor_tau: int = 1,
                 clamp_min: float = 0.0, clamp_max: float = 1.0) -> torch.Tensor:
    """
    Denoising Projected Gradient Descent.
 
    At each iteration:
      1. Take a standard PGD step.
      2. Check the logit-margin constraint: -Phi(x_t) >= c  (i.e. the example
         already fools the model with margin >= c).
      3. If satisfied `control_factor_tau` times cumulatively (the `con`
         counter), purify the current adversarial example via one diffuse+
         denoise step before continuing to attack.
 
    Args:
        margin_c: minimum required attack strength (paper's `c` in Eq. 8).
                  0.0 is the natural default: any successful flip counts.
        control_factor_tau: how many times the constraint must be satisfied
                  before purification triggers (paper's `tau`; tau=1 used
                  throughout their main experiments, per their sensitivity
                  analysis in Table 9).
    """
    x0 = x.clone().detach()
    delta = torch.zeros_like(x)
    con = torch.zeros(x.size(0), device=x.device)
 
    for t in range(steps):
        x_t = torch.clamp(x0 + delta, clamp_min, clamp_max).detach().requires_grad_(True)
        logits = model(x_t)
 
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
 
 
# ---------------------------------------------------------------------------
# 4. DifAT loss (Eq. 11): min over relatively-weak DPGD examples, not the
#    strongest possible ones
# ---------------------------------------------------------------------------
 
def difat_loss(model: nn.Module, x_adv: torch.Tensor, y: torch.Tensor,
                criterion) -> torch.Tensor:
    """
    DifAT trains directly on DPGD-purified adversarial examples with plain
    cross-entropy -- the "weak/appropriate-intensity" selection already
    happened inside dpgd_attack via the purification step, so no extra
    reweighting is needed here (unlike the FGSM margin-weighting experiment).
    """
    logits_adv = model(x_adv)
    return criterion(logits_adv, y), logits_adv
 
