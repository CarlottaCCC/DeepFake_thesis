import torch
import torch.nn.functional as F
from utils import *

def nes_attack(
    model,
    x,
    target_label,
    epsilon=8/255,
    alpha=2/255,
    sigma=1e-3,
    nes_samples=20,
    nes_iters=10,
    clip_min=0.0,
    clip_max=1.0,
    device="cuda"
):
    """
    NES-based black-box attack (L∞ constrained)

    Args:
        model: trained classifier (outputs logits or probabilities)
        x: input tensor (1,C,H,W)
        target_label: int (target class index)
        epsilon: L∞ bound
        alpha: step size
        sigma: noise scale for NES
        nes_samples: number of noise samples per iteration
        nes_iters: number of attack iterations
    """

    model.eval()
    x = x.to(device)
    x_adv = x.clone().detach()

    # Keep original for projection
    x_orig = x.clone().detach()

    for i in range(nes_iters):

        grad_estimate = torch.zeros_like(x_adv)

        for _ in range(nes_samples):

            noise = torch.randn_like(x_adv)

            x_pos = x_adv + sigma * noise
            x_neg = x_adv - sigma * noise

            with torch.no_grad():
                logits_pos = model(x_pos)
                logits_neg = model(x_neg)

                #probs_pos = F.softmax(logits_pos, dim=1)
                #probs_neg = F.softmax(logits_neg, dim=1)

                ## Targeted attack: maximize target prob
                #loss_pos = -torch.log(probs_pos[:, target_label] + 1e-8)
                #loss_neg = -torch.log(probs_neg[:, target_label] + 1e-8)
                loss_pos = F.cross_entropy(logits_pos, target_label, reduction='none')
                loss_neg = F.cross_entropy(logits_neg, target_label, reduction='none')

            grad_estimate += (loss_pos - loss_neg).view(-1,1,1,1) * noise

        grad_estimate = grad_estimate / (2 * sigma * nes_samples)

        # L∞ update
        x_adv = x_adv - alpha * torch.sign(grad_estimate)

        # Projection
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv.detach()

