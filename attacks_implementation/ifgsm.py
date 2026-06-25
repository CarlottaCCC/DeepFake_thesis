import torch
import torch.nn.functional as F

def ifgsm_attack(images, labels, epsilon, alpha, num_iter, model, criterion):
    """
    images:   batch di immagini (B, C, H, W)
    labels:   label del batch   (B,)
    epsilon:  perturbazione massima
    alpha:    step size per ogni iterazione
    num_iter: numero di iterazioni
    """
    perturbed_images = images.clone().detach()

    for i in range(num_iter):
        perturbed_images.requires_grad_(True)

        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            sign_grad = perturbed_images.grad.sign()
            perturbed_images = perturbed_images + alpha * sign_grad

            # Proiezione nella epsilon-ball attorno all'immagine originale
            perturbed_images = torch.clamp(perturbed_images, images - epsilon, images + epsilon)

            # Clip nel range valido [0, 1]
            perturbed_images = torch.clamp(perturbed_images, 0, 1)

            # Fondamentale: stacca dal grafo per la prossima iterazione
            perturbed_images = perturbed_images.detach()

    return perturbed_images