import torch
from tqdm import tqdm

def nes_attack_first_v(model, x, y, eps=0.05, sigma=0.001,
               n_samples=100, step_size=0.01, n_iters=100,
               targeted=False, device='cuda'):
    
    x_adv = x.clone().to(device)
    x_orig = x.clone().to(device)
    y = y.to(device)
    model = model.to(device)

    loop = tqdm(range(n_iters), desc="")

    #for _ in range(n_iters):
    for _ in loop:
        grad_est = torch.zeros_like(x_adv)

        for _ in range(n_samples // 2):
            delta = torch.randn_like(x_adv)  # automatically on same device as x_adv

            x_pos = (x_adv + sigma * delta).clamp(0, 1)
            x_neg = (x_adv - sigma * delta).clamp(0, 1)

            with torch.no_grad():
                loss_pos = torch.nn.functional.cross_entropy(model(x_pos), y)
                loss_neg = torch.nn.functional.cross_entropy(model(x_neg), y)

            grad_est += (loss_pos - loss_neg) * delta

        grad_est /= (2 * sigma * (n_samples // 2))

        if targeted:
            x_adv = x_adv - step_size * grad_est.sign()
        else:
            x_adv = x_adv + step_size * grad_est.sign()

        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0, 1)

    return x_adv

def nes_attack(model, x, y, eps=8/255, sigma=0.001,
               n_samples=100, step_size=2/255, n_iters=50,
               targeted=False, device='cuda'):

    x_adv = x.clone().to(device)
    x_orig = x.clone().to(device)
    y = y.to(device)
    model = model.to(device)

    for _ in tqdm(range(n_iters), desc=""):
        grad_est = torch.zeros_like(x_adv)

        for _ in range(n_samples // 2):
            delta = torch.randn_like(x_adv)

            x_pos = (x_adv + sigma * delta).clamp(0, 1)
            x_neg = (x_adv - sigma * delta).clamp(0, 1)

            with torch.no_grad():
                loss_pos = torch.nn.functional.cross_entropy(model(x_pos), y, reduction='none')  # (N,)
                loss_neg = torch.nn.functional.cross_entropy(model(x_neg), y, reduction='none')  # (N,)

            loss_diff = (loss_pos - loss_neg).view(-1, 1, 1, 1)  # per-example scalar
            grad_est += loss_diff * delta   # each image weighted by ITS OWN loss change

        grad_est /= (2 * sigma * (n_samples // 2))

        if targeted:
            x_adv = x_adv - step_size * grad_est.sign()
        else:
            x_adv = x_adv + step_size * grad_est.sign()

        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0, 1)

    return x_adv