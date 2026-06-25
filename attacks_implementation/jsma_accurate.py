import torch
import torch.nn.functional as F

def compute_jacobian(model, image, nb_classes):
    """
    Calcola il Jacobiano completo in un unico passaggio.
    Equivalente a cleverhans.attacks.jacobian() in TF.
    Ritorna: tensore [nb_classes, C, H, W]
    """
    grads = []
    for cls in range(nb_classes):
        img = image.detach().requires_grad_(True)
        logits = model(img)
        model.zero_grad()
        logits[0, cls].backward()
        grads.append(img.grad.detach().clone())  # [1, C, H, W]
    return torch.stack(grads, dim=0)  # [nb_classes, 1, C, H, W]


def wjsma_binary(model, image, target_class,
                 max_iter=100, theta=1/255,
                 clip_min=0.0, clip_max=1.0):
    """
    WJSMA fedele al paper Combey et al. 2020.
    
    Saliency WJSMA (eq. paper):
        alpha_i = p_t(x) * dF_t/dx_i
        beta_i  = p_o(x) * dF_o/dx_i
        S_i = alpha_i * |beta_i|  se alpha_i > 0 e beta_i < 0, else 0
    """
    model.eval()
    nb_classes = 2
    other_class = 1 - target_class

    adv = image.unsqueeze(0).clone().detach()

    # Search domain: pixel non ancora saturati (modificabili)
    # shape: [1, C, H, W] - True = ancora modificabile
    search_domain = torch.ones_like(adv, dtype=torch.bool)

    for _ in range(max_iter):
        # Check predizione corrente
        with torch.no_grad():
            logits = model(adv)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()

        if pred == target_class:
            break

        # Jacobiano completo: [nb_classes, 1, C, H, W]
        J = compute_jacobian(model, adv, nb_classes)

        grad_target = J[target_class].squeeze(0)   # [C, H, W]
        grad_other  = J[other_class].squeeze(0)    # [C, H, W]

        p_t = probs[0, target_class].item()
        p_o = probs[0, other_class].item()

        # WJSMA: pesa i gradienti per le probabilità
        alpha = p_t * grad_target   # [C, H, W]
        beta  = p_o * grad_other    # [C, H, W]

        # Mask: solo pixel con alpha > 0 e beta < 0
        mask = (alpha > 0) & (beta < 0) & search_domain.squeeze(0)
        saliency = alpha * torch.abs(beta)
        saliency[~mask] = 0

        if saliency.max() == 0:
            break  # nessun pixel utile rimasto

        # Seleziona il pixel con saliency massima
        idx = saliency.view(-1).argmax()
        c, h, w = torch.unravel_index(idx, saliency.shape)

        # Applica perturbazione
        new_val = torch.clamp(adv[0, c, h, w] + theta, clip_min, clip_max)
        adv[0, c, h, w] = new_val

        # Rimuovi dalla search domain se saturato
        if new_val >= clip_max or new_val <= clip_min:
            search_domain[0, c, h, w] = False

    return adv.squeeze(0)


def tjsma_binary(model, image, target_class,
                 max_iter=100, theta=1/255,
                 clip_min=0.0, clip_max=1.0):
    """
    TJSMA fedele al paper Combey et al. 2020.
    
    Saliency TJSMA (eq. paper):
        alpha_i = x_i * p_t(x) * dF_t/dx_i       ← pesa per il valore del pixel
        beta_i  = x_i * p_o(x) * dF_o/dx_i
        S_i = alpha_i * |beta_i|  se alpha_i > 0 e beta_i < 0, else 0
    
    La differenza con WJSMA è il termine x_i che penalizza i pixel
    già vicini all'estremo (feature extremal penalization).
    """
    model.eval()
    nb_classes = 2
    other_class = 1 - target_class

    adv = image.unsqueeze(0).clone().detach()
    search_domain = torch.ones_like(adv, dtype=torch.bool)

    for _ in range(max_iter):
        with torch.no_grad():
            logits = model(adv)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()

        if pred == target_class:
            break

        J = compute_jacobian(model, adv, nb_classes)

        grad_target = J[target_class].squeeze(0)
        grad_other  = J[other_class].squeeze(0)

        p_t = probs[0, target_class].item()
        p_o = probs[0, other_class].item()

        x = adv.squeeze(0)  # [C, H, W] - valore corrente del pixel

        # TJSMA: moltiplica anche per x_i (extremal penalization)
        alpha = x * p_t * grad_target
        beta  = x * p_o * grad_other

        mask = (alpha > 0) & (beta < 0) & search_domain.squeeze(0)
        saliency = alpha * torch.abs(beta)
        saliency[~mask] = 0

        if saliency.max() == 0:
            break

        idx = saliency.view(-1).argmax()
        c, h, w = torch.unravel_index(idx, saliency.shape)

        new_val = torch.clamp(adv[0, c, h, w] + theta, clip_min, clip_max)
        adv[0, c, h, w] = new_val

        if new_val >= clip_max or new_val <= clip_min:
            search_domain[0, c, h, w] = False

    return adv.squeeze(0)

def wjsma_binary_debug(model, image, target_class,
                        max_iter=100, theta=0.05,
                        clip_min=0.0, clip_max=1.0):
    model.eval()
    nb_classes = 2
    other_class = 1 - target_class
    adv = image.unsqueeze(0).clone().detach()
    search_domain = torch.ones_like(adv, dtype=torch.bool)

    for i in range(max_iter):
        with torch.no_grad():
            logits = model(adv)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()

        if pred == target_class:
            print(f"Successo all'iterazione {i}")
            break

        J = compute_jacobian(model, adv, nb_classes)
        grad_target = J[target_class].squeeze(0)
        grad_other  = J[other_class].squeeze(0)

        p_t = probs[0, target_class].item()
        p_o = probs[0, other_class].item()

        alpha = p_t * grad_target
        beta  = p_o * grad_other

        mask = (alpha > 0) & (beta < 0) & search_domain.squeeze(0)
        saliency = alpha * torch.abs(beta)
        saliency[~mask] = 0

        # ← DEBUG
        print(f"[iter {i:03d}] pred={pred} p_t={p_t:.4f} p_o={p_o:.4f} "
              f"valid_pixels={mask.sum().item()} "
              f"max_saliency={saliency.max().item():.6f} "
              f"alpha_pos={( alpha>0).sum().item()} "
              f"beta_neg={(beta<0).sum().item()}")

        if saliency.max() == 0:
            print("Saliency tutta zero — attacco bloccato")
            break

        idx = saliency.view(-1).argmax()
        c, h, w = torch.unravel_index(idx, saliency.shape)
        new_val = torch.clamp(adv[0, c, h, w] + theta, clip_min, clip_max)
        adv[0, c, h, w] = new_val

        if new_val >= clip_max or new_val <= clip_min:
            search_domain[0, c, h, w] = False

    return adv.squeeze(0)
''' 

Esegui su **una singola immagine** e dimmi cosa stampa. I casi possibili sono:

**Caso A — `valid_pixels` crolla a 0 rapidamente:**
```
[iter 000] valid_pixels=45231 max_saliency=0.0023
[iter 001] valid_pixels=45230 max_saliency=0.0021
...
[iter 050] valid_pixels=0    → ⛔ Saliency tutta zero
```
→ La search domain si esaurisce, `theta` troppo grande o immagine già vicina ai bordi

**Caso B — `valid_pixels` alto ma `max_saliency` piccolissima:**
```
[iter 000] valid_pixels=80000 max_saliency=0.000001
```
→ I gradienti sono quasi zero, il modello è insensibile alle perturbazioni pixel-wise (usa BatchNorm o feature globali)

**Caso C — `p_t` non cresce mai nonostante le modifiche:**
```
[iter 000] p_t=0.05  p_o=0.95
[iter 050] p_t=0.06  p_o=0.94  ← non si muove
'''