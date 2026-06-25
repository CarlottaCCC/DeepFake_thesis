import torch

def nes_attack(model, x, y, eps=0.05, sigma=0.001,
               n_samples=100, step_size=0.01, n_iters=100,
               targeted=False, device='cuda'):
    
    x_adv = x.clone().to(device)
    x_orig = x.clone().to(device)
    y = y.to(device)
    model = model.to(device)

    for _ in range(n_iters):
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