import torch
import torch.nn.functional as F

def wjsma_binary(model, image, target_class, max_iter=100, gamma=0.01):
    model.eval()
    # ✅ Fix 1: requires_grad_ dopo detach
    image = image.unsqueeze(0).clone().detach().requires_grad_(True)
    
    for _ in range(max_iter):
        # ✅ Fix 2: ricrea il tensore ogni iterazione per grafo pulito
        image = image.detach().requires_grad_(True)
        
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        
        if pred == target_class:
            break
        
        other_class = 1 - target_class
        
        # ✅ Fix 5: due forward pass separati per evitare problemi col grafo
        logits_t = model(image)
        logits_t[0, target_class].backward(retain_graph=True)
        grad_target = image.grad.detach().clone()
        
        image.grad.zero_()
        
        logits_o = model(image)
        logits_o[0, other_class].backward()
        grad_other = image.grad.detach().clone()
        
        # ✅ Fix 4: alpha con termine di distanza (come TJSMA)
        alpha = (1 - image.detach()) * grad_target
        beta  = (1 - image.detach()) * probs[0, other_class].detach() * grad_other
        
        mask = (alpha > 0) & (beta < 0)
        saliency = alpha * torch.abs(beta)
        saliency[~mask] = 0
        
        # ✅ Fix 3: lavora senza batch dimension
        saliency_nobatch = saliency.squeeze(0)
        idx = saliency_nobatch.view(-1).argmax()
        c, h, w = torch.unravel_index(idx, saliency_nobatch.shape)
        
        with torch.no_grad():
            image[0, c, h, w] = torch.clamp(image[0, c, h, w] + gamma, 0, 1)
    
    return image.detach().squeeze(0)


def tjsma_binary(model, image, target_class, max_iter=100, gamma=0.01):
    model.eval()
    image = image.unsqueeze(0).clone().detach().requires_grad_(True)
    
    for _ in range(max_iter):
        image = image.detach().requires_grad_(True)
        
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        
        if pred == target_class:
            break
        
        other_class = 1 - target_class
        
        logits_t = model(image)
        logits_t[0, target_class].backward(retain_graph=True)
        grad_target = image.grad.detach().clone()
        
        image.grad.zero_()
        
        logits_o = model(image)
        logits_o[0, other_class].backward()
        grad_other = image.grad.detach().clone()
        
        alpha = (1 - image.detach()) * grad_target
        beta  = (1 - image.detach()) * probs[0, other_class].detach() * grad_other
        
        mask = (alpha > 0) & (beta < 0)
        saliency = alpha * torch.abs(beta)
        saliency[~mask] = 0
        
        saliency_nobatch = saliency.squeeze(0)
        idx = saliency_nobatch.view(-1).argmax()
        c, h, w = torch.unravel_index(idx, saliency_nobatch.shape)
        
        with torch.no_grad():
            image[0, c, h, w] = torch.clamp(image[0, c, h, w] + gamma, 0, 1)
    
    return image.detach().squeeze(0)