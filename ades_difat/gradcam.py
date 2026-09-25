import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import statistics
from torchvision import transforms
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from utils import *
from attacks import *
import itertools
import foolbox as fb
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from test_attacks import test_attack
from training_ades_difat import standard_pgd_attack
from attacks_implementation.genattack import GenAttack
from attacks_implementation.nes import nes_attack, nes_attack_first_v
from pathlib import Path

def save_gradcam_samples(test_loader, save_path, n_per_class=5):
    real_samples = []
    fake_samples = []

    for imgs, labels, idx in test_loader:

        for i in range(len(labels)):

            label = labels[i].item()

            if label == 0 and len(real_samples) < n_per_class:
                real_samples.append({
                    "image": imgs[i].clone(),
                    "label": labels[i].clone(),
                    "idx": idx[i]
                })

            elif label == 1 and len(fake_samples) < n_per_class:
                fake_samples.append({
                    "image": imgs[i].clone(),
                    "label": labels[i].clone(),
                    "idx": idx[i]
                })

        if len(real_samples) >= n_per_class and len(fake_samples) >= n_per_class:
            break

    if len(real_samples) < n_per_class:
        raise RuntimeError(
            f"Could not find {n_per_class} real samples, only found {len(real_samples)}."
        )

    if len(fake_samples) < n_per_class:
        raise RuntimeError(
            f"Could not find {n_per_class} fake samples, only found {len(fake_samples)}."
        )

    torch.save(
        {
            "real": real_samples,
            "fake": fake_samples
        },
        save_path
    )

    print(f"Saved {len(real_samples)} real and {len(fake_samples)} fake samples to: {save_path}")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, x, target_class):
        """
        x:
            [B, 3, H, W], already normalized

        target_class:
            [B] tensor containing the class whose Grad-CAM
            we want to compute.
        """

        self.model.zero_grad(set_to_none=True)

        # Forward
        logits = self.model(x)

        # Select the score of the desired class
        target_scores = logits[
            torch.arange(
                logits.size(0),
                device=logits.device
            ),
            target_class
        ]

        # Backpropagate only the selected class
        target_scores.sum().backward()

        activations = self.activations
        gradients = self.gradients

        print("\nTarget class:", target_class)
        print("Logits:", logits)
        
        print("\nActivations:")
        print("min:", activations.min().item())
        print("max:", activations.max().item())
        print("mean:", activations.mean().item())
        
        print("\nGradients:")
        print("min:", gradients.min().item())
        print("max:", gradients.max().item())
        print("mean:", gradients.mean().item())
        print("abs mean:", gradients.abs().mean().item())
        print("nonzero:", (gradients != 0).float().mean().item())

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted activation maps
        cam_raw = (weights * activations).sum(
            dim=1,
            keepdim=True
        )

        print("\nWeights:")
        print("min:", weights.min().item())
        print("max:", weights.max().item())
        print("mean:", weights.mean().item())
        print("abs mean:", weights.abs().mean().item())
        
        print("\nCAM RAW:")
        print("min:", cam_raw.min().item())
        print("max:", cam_raw.max().item())
        print("mean:", cam_raw.mean().item())
        print("abs mean:", cam_raw.abs().mean().item())
        print("positive fraction:", (cam_raw > 0).float().mean().item())
        print("negative fraction:", (cam_raw < 0).float().mean().item())
        negative_fraction = (cam_raw < 0).float().mean().item()

        # ReLU
        #if negative_fraction == 1.0:
        #    cam = F.relu(-cam_raw)
        cam = F.relu(cam_raw)

        # Resize to image resolution
        cam = F.interpolate(
            cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Normalize each CAM independently
        B = cam.shape[0]

        cam_flat = cam.view(B, -1)

        cam_min = cam_flat.min(
            dim=1,
            keepdim=True
        )[0]

        cam_max = cam_flat.max(
            dim=1,
            keepdim=True
        )[0]

        cam_flat = (
            cam_flat - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        cam = cam_flat.view(
            B,
            1,
            x.shape[-2],
            x.shape[-1]
        )

        return cam.detach(), logits.detach()


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
)


def denormalize(x):
    """
    x: [B, 3, H, W]
    """

    mean = IMAGENET_MEAN.to(
        device=x.device,
        dtype=x.dtype
    ).view(1, 3, 1, 1)

    std = IMAGENET_STD.to(
        device=x.device,
        dtype=x.dtype
    ).view(1, 3, 1, 1)

    return x * std + mean


def tensor_to_image(x):
    """
    x: [3, H, W]
    """

    x = x.detach().cpu()

    x = x.permute(1, 2, 0).numpy()

    return np.clip(x, 0, 1)

def plot_gradcam_clean(
        clean_raw,
        clean_cam,
        true_class,
        clean_logits,
        save_path=None
):
    clean_img = tensor_to_image(clean_raw[0])
    
    clean_cam_np = clean_cam[0, 0].cpu().numpy()

    clean_prob = F.softmax(
        clean_logits,
        dim=1
    )[0]

    clean_pred = clean_logits.argmax(
        dim=1
    )[0].item()

    clean_conf = clean_prob[
        clean_pred
    ].item()

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(clean_img)
    ax.imshow(clean_cam_np, cmap="jet", alpha=0.45)
    
    ax.set_title(
        f"Clean\n"
        f"pred={clean_pred}, "
        f"conf={clean_conf:.3f}"
    )
    ax.axis("off")

    fig.suptitle(
        f"Grad-CAM analysis — true class = {true_class}",
        fontsize=20
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM attack grid to {save_path}")
      

def plot_gradcam_grid(
    attacked_raw,
    attacked_cams,
    attack_names,
    epsilons,
    true_class,
    attacked_logits,
    save_path=None
):
    """
    clean_raw:
        [1, 3, H, W]

    attacked_raw:
        dict:
            {
                "FGSM": [3, 1, 3, H, W],
                "IFGSM": [3, 1, 3, H, W],
                "PGD": [3, 1, 3, H, W]
            }

    clean_cam:
        [1, 1, H, W]

    attacked_cams:
        dict:
            {
                "FGSM": [3, 1, H, W],
                ...
            }

    epsilons:
        list of epsilon values

    true_class:
        integer
    """

    n_attacks = len(attack_names)
    n_eps = len(epsilons)

    # One clean column + 9 attack columns
    n_cols = n_eps

    # Clean row + one row per attack
    n_rows = n_attacks

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows)
    )

    # ---------------------------------------------------------
    # Attacks
    # ---------------------------------------------------------

    for row, attack_name in enumerate(
        attack_names,
        start=0
    ):

        # Label attack on left
        axes[row, 0].text(
            -0.15, 0.5,           # x < 0 pushes it outside the axes, to the left
            attack_name,
            ha="right",
            va="center",
            fontsize=18,
            rotation=0,
            transform=axes[row, 0].transAxes
        )

        axes[row, 0].axis("off")

        for j, eps in enumerate(epsilons):

            attacked_img = tensor_to_image(
                attacked_raw[attack_name][j, 0]
            )

            cam_np = attacked_cams[
                attack_name
            ][j, 0].cpu().numpy()

            logits = attacked_logits[attack_name][j]

            probs = torch.softmax(logits, dim=0)

            pred = logits.argmax().item()

            conf = probs[pred].item()

            ax = axes[
                row,
                j
            ]

            ax.imshow(attacked_img)

            ax.imshow(
                cam_np,
                cmap="jet",
                alpha=0.45
            )

            eps_255 = eps * 255

            ax.set_title(
                f"ε={eps_255:.0f}/255\n"
                f"pred={pred}, "
                f"conf={conf:.3f}"
            )

            ax.axis("off")

    fig.suptitle(
        f"Grad-CAM analysis — true class = {true_class}",
        fontsize=20
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM attack grid to {save_path}")

def generate_attack_grid(
    model,
    normalize,
    x_raw,
    y
):

    attacked_raw = {}
    attacked_logits = {}

    attack_names = ["FGSM", "IFGSM", "PGD"]
    epsilons = [2/255, 4/255, 8/255]

    # attacks functions
    fgsm = fb.attacks.FGSM()
    ifgsm = fb.attacks.LinfPGD(
            steps=20,
            abs_stepsize=1/255,
            random_start=False)

    for attack_name in attack_names:

        attacked_raw[attack_name] = []
        attacked_logits[attack_name] = []

    # ---------------------------------------------------------
    # Generate all attacks
    # ---------------------------------------------------------

    preprocessing = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        axis=-3  # per PyTorch (C, H, W)
        )
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing, device=device)

    for eps in epsilons:

        # FGSM
        _, adv, _ = fgsm(fmodel, x_raw, y, epsilons=eps)

        attacked_raw["FGSM"].append(
            adv.detach()
        )

        # IFGSM
        _, adv, _ = ifgsm(fmodel, x_raw, y, epsilons=eps)

        attacked_raw["IFGSM"].append(
            adv.detach()
        )

        # PGD
        adv = standard_pgd_attack(model, x_raw.detach(), 
                                       y, eps, alpha=2/255, 
                                       steps=20, normalize=normalize)

        attacked_raw["PGD"].append(
            adv.detach()
        )

    # Convert lists to tensors
    for attack_name in attack_names:

        attacked_raw[attack_name] = torch.stack(
            attacked_raw[attack_name]
        )
    return attacked_raw

def select_sample_indices(test_dataset, n_samples, label, seed=42):

    label_map = {
        "real": 0,
        "fake": 1
    }

    if label not in label_map:
        raise ValueError("label must be 'real' or 'fake'")

    target_label = label_map[label]

    matching_indices = []

    for idx in range(len(test_dataset)):
        _, sample_label, _ = test_dataset[idx]

        if sample_label == target_label:
            matching_indices.append(idx)

    if len(matching_indices) < n_samples:
        raise ValueError(
            f"Requested {n_samples} {label} samples, "
            f"but only found {len(matching_indices)}."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    selected_positions = torch.randperm(
        len(matching_indices),
        generator=generator
    )[:n_samples]

    return [
        matching_indices[i]
        for i in selected_positions.tolist()
    ]

def run_gradcam_analysis(
    model,
    normalize,
    gradcam,
    x_raw,
    y,
    attacked_raw,
    attack_names,
    epsilons,
    attacked_save_path,
    original_save_path
):

    model.eval()

    # ---------------------------------------------------------
    # Clean Grad-CAM
    # ---------------------------------------------------------

    x = normalize(x_raw)

    clean_cam, clean_logits = gradcam(
        x,
        target_class=y
    )

    # ---------------------------------------------------------
    # Grad-CAM for every attacked image
    # ---------------------------------------------------------

    attacked_cams = {}
    attacked_logits = {}

    for attack_name in attack_names:

        attacked_cams[attack_name] = []
        attacked_logits[attack_name] = []

        for j in range(len(epsilons)):

            adv_raw = attacked_raw[attack_name][j]

            adv = normalize(adv_raw)

            cam, logits = gradcam(adv,target_class=y)

            attacked_cams[attack_name].append(cam)

            attacked_logits[attack_name].append(logits[0])

    # Stack
    for attack_name in attack_names:

        attacked_cams[attack_name] = torch.cat(
            attacked_cams[attack_name],
            dim=0
        )

        attacked_logits[attack_name] = torch.stack(
            attacked_logits[attack_name]
        )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plot_gradcam_clean(
        clean_raw=x_raw,
        clean_cam=clean_cam,
        true_class=y.item(),
        clean_logits=clean_logits,
        save_path=original_save_path
    )


    plot_gradcam_grid(
        attacked_raw=attacked_raw,
        attacked_cams=attacked_cams,
        attack_names=attack_names,
        epsilons=epsilons,
        true_class=y.item(),
        attacked_logits=attacked_logits,
        save_path=attacked_save_path
    )
    

def show_gradcam_comparison(
    clean_raw,
    adv_raw,
    clean_cam,
    adv_cam,
    clean_pred,
    adv_pred,
    clean_conf,
    adv_conf,
    epsilon,
    save_path,
    idx=0
):

    clean_img = clean_raw[idx].detach().cpu().permute(1, 2, 0).numpy()
    adv_img = adv_raw[idx].detach().cpu().permute(1, 2, 0).numpy()

    clean_img = np.clip(clean_img, 0, 1)
    adv_img = np.clip(adv_img, 0, 1)

    clean_heatmap = clean_cam[idx, 0].cpu().numpy()
    adv_heatmap = adv_cam[idx, 0].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Clean image
    axes[0, 0].imshow(clean_img)
    axes[0, 0].set_title(
        f"Clean\n"
        f"Pred={clean_pred[idx].item()} "
        f"Conf={clean_conf[idx].item():.3f}"
    )
    axes[0, 0].axis("off")

    # Adversarial image
    axes[0, 1].imshow(adv_img)
    axes[0, 1].set_title(
        f"Adversarial\n"
        f"Pred={adv_pred[idx].item()} "
        f"Conf={adv_conf[idx].item():.3f}\n"
        f"ε={epsilon}"
    )
    axes[0, 1].axis("off")

    # Clean GradCAM
    axes[1, 0].imshow(clean_img)
    axes[1, 0].imshow(
        clean_heatmap,
        cmap="jet",
        alpha=0.45
    )
    axes[1, 0].set_title("Clean Grad-CAM")
    axes[1, 0].axis("off")

    # Adversarial GradCAM
    axes[1, 1].imshow(adv_img)
    axes[1, 1].imshow(
        adv_heatmap,
        cmap="jet",
        alpha=0.45
    )
    axes[1, 1].set_title("Adversarial Grad-CAM")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM attack grid to {save_path}")


def plot_gradcam_grid_across_models(
    models_data,
    attack_name,
    epsilons,
    eps,
    true_class,
    n_cols=None,
    save_path=None,
    save_path_clean=None
):
    """
    Grid of Grad-CAMs across models, for a single attack at a single epsilon.

    models_data:
        dict: {
            model_name: {
                "raw":    {attack_name: [n_eps, batch, 3, H, W]},
                "cams":   {attack_name: [n_eps, batch, H, W]},
                "logits": {attack_name: [n_eps, batch, num_classes]}
            }
        }

    attack_name:
        str, e.g. "PGD" — must be a key in each model's dicts

    epsilons:
        the same epsilons list used in generate_attack_grid, used to find eps's index

    eps:
        float, the specific epsilon value to plot (must be in epsilons)

    true_class:
        integer
    """

    # ATTACKED IMGS
    eps_idx = epsilons.index(eps)  # raises ValueError if eps not found — good, fail loud

    model_names = list(models_data.keys())
    n_models = len(model_names)

    if n_cols is None:
        n_cols = math.ceil(math.sqrt(n_models))
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 3 * n_rows)
    )
    axes = np.array(axes).reshape(-1)

    eps_255 = eps * 255

    for i, model_name in enumerate(model_names):

        data = models_data[model_name]

        raw_img = tensor_to_image(
            data["raw"][attack_name][eps_idx, 0]
        )

        cam_np = data["cams"][attack_name][eps_idx, 0]
        if hasattr(cam_np, "cpu"):
            cam_np = cam_np.cpu().numpy()
        cam_np = np.squeeze(cam_np)

        logits = data["logits"][attack_name][eps_idx]
        if logits.ndim > 1:  # in case batch dim survived
            logits = logits[0]

        probs = torch.softmax(logits, dim=0)
        pred = logits.argmax().item()
        conf = probs[pred].item()

        ax = axes[i]
        ax.imshow(raw_img)
        ax.imshow(cam_np, cmap="jet", alpha=0.45)

        color = "green" if pred == true_class else "red"
        ax.set_title(
            f"{model_name}\npred={pred}, conf={conf:.3f}",
            fontsize=13, color=color
        )
        ax.axis("off")

    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Grad-CAM across models — attack={attack_name}, "
        f"ε={eps_255:.0f}/255, true class = {true_class}",
        fontsize=18
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved model-comparison Grad-CAM grid to {save_path}")

def plot_clean_gradcam_grid_across_models(
    models_data,
    true_class,
    n_cols=None,
    save_path_clean=None):

    # Plotting the gradcams for pre-attacked imgs

    model_names = list(models_data.keys())
    n_models = len(model_names)

    if n_cols is None:
        n_cols = math.ceil(math.sqrt(n_models))
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 3 * n_rows)
    )
    axes = np.array(axes).reshape(-1)
    

    for i, model_name in enumerate(model_names):
    
        data = models_data[model_name]

        raw_img = tensor_to_image(
            data["clean_img"]
        )

        cam_np = data["clean_cam"][0, 0].cpu().numpy()
        clean_logits = data["clean_logit"]
        
        prob = F.softmax(
            clean_logits,
            dim=1
        )[0]
    
        pred = clean_logits.argmax(
            dim=1
        )[0].item()
    
        conf = prob[
            pred
        ].item()
        if hasattr(cam_np, "cpu"):
            cam_np = cam_np.cpu().numpy()
        cam_np = np.squeeze(cam_np)

    ######################
        #logits = data["clean_logit"]
        #if logits.ndim > 1:  # in case batch dim survived
        #    logits = logits[0]
#
        #probs = torch.softmax(logits, dim=0)
        #pred = logits.argmax().item()
        #conf = probs[pred].item()

        ax = axes[i]
        ax.imshow(raw_img)
        ax.imshow(cam_np, cmap="jet", alpha=0.45)

        color = "green" if pred == true_class else "red"
        ax.set_title(
            f"{model_name}\npred={pred}, conf={conf:.3f}",
            fontsize=13, color=color
        )
        ax.axis("off")

    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Grad-CAM across models — pre-attack"
        f"true class = {true_class}",
        fontsize=18
    )

    plt.tight_layout()

    if save_path_clean is not None:
        save_path_clean = Path(save_path_clean)
        save_path_clean.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_clean, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved model-comparison Grad-CAM grid to {save_path_clean}")
    

def build_models_data(
    models,          # dict: {model_name: model}
    normalize,
    x_raw,
    y,
    gradcam       # your existing function that computes a CAM for one image+model
):
    """
    Populates the models_data dict expected by plot_gradcam_grid_across_models.

    models:
        dict {model_name: model_object}

    gradcam_fn:
        callable(model, img_tensor, target_class) -> cam [H, W]
        (whatever your existing Grad-CAM function's actual signature is —
        adjust the call below to match it)
    """

    models_data = {}

    for models_values in models:

        model_label = models_values["model_label"]
        model_name = models_values["model_name"]

        model_path = f"{MODELS_DIR}/{model_name}"

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(model.fc.in_features, 2)
        )
        model = model.to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        del checkpoint 

        # I compute the clean cams
        clean_img = normalize(x_raw)
        clean_cam, clean_logits = gradcam(clean_img, y)

        # 1) Generate attacked images for this model
        attacked_raw = generate_attack_grid(model, normalize, x_raw, y)

        attack_names = list(attacked_raw.keys())

        attacked_cams = {}
        attacked_logits = {}

        # 2) For each attack, each epsilon slice, compute cam + logits
        for attack_name in attack_names:

            n_eps = attacked_raw[attack_name].shape[0]
            cams_list = []
            logits_list = []

            for eps_idx in range(n_eps):

                img = attacked_raw[attack_name][eps_idx]  # [batch, 3, H, W]

                cam, logits = gradcam(img, y)
                cams_list.append(cam.detach().cpu())       # detach + move off GPU
                logits_list.append(logits.cpu())

                model.zero_grad(set_to_none=True)          # clear any grads gradcam left behind

            attacked_cams[attack_name] = torch.stack(cams_list)
            attacked_logits[attack_name] = torch.stack(logits_list)

        # move raw images off GPU too — only needed on CPU for plotting later
        attacked_raw = {
            k: v.detach().cpu() for k, v in attacked_raw.items()
        }

        models_data[model_label] = {
            "clean_img": clean_img[0].detach().cpu(),      # [3, H, W]
            "clean_cam": clean_cam.cpu(),
            "clean_logit": clean_logits.cpu(),
            "raw": attacked_raw,
            "cams": attacked_cams,
            "logits": attacked_logits
        }

        del model
        torch.cuda.empty_cache()

    return models_data

def build_clean_models_data(
    models,
    normalize,
    x_raw,
    y,
    gradcam
):
    """
    Computes clean (unattacked) Grad-CAMs for each model on the same x_raw.

    Returns:
        dict {model_label: {"raw": tensor[3,H,W], "cam": tensor[H,W], "logits": tensor[num_classes]}}
    """

    models_data_clean = {}

    for models_values in models:

        model_label = models_values["model_label"]
        model_name = models_values["model_name"]
        model_path = f"{MODELS_DIR}/{model_name}"

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(model.fc.in_features, 2)
        )
        model = model.to(device)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        del checkpoint

        img = normalize(x_raw)
        cam, logits = gradcam(img, y)

        models_data_clean[model_label] = {
            "raw": x_raw[0].detach().cpu(),      # [3, H, W]
            "cam": cam.cpu(),
            "logits": logits.cpu()
        }

        model.zero_grad(set_to_none=True)
        del model
        torch.cuda.empty_cache()

    return models_data_clean



def build_gradcams(samples):
    attack_names = ["FGSM", "IFGSM", "PGD"]
    epsilons = [2/255, 4/255, 8/255]
    
    #print(samples_list)
    # REAL IMAGES
    real_list = samples["real"]
    model.eval()
    for i, real in enumerate(real_list):
        #print(f"Analyzing image {i}")
    
        real_raw = real["image"].to(device).unsqueeze(0)
        real_label = real["label"].to(device).unsqueeze(0)
        attacked_real = generate_attack_grid(model, normalize, real_raw, real_label)
        target_layer = model.layer4[-1].conv3
            
        gradcam = GradCAM(
            model,
            target_layer
        )
        models_data = build_models_data(models_names, normalize, real_raw, real_label, gradcam=gradcam)

        print("Building gradcam for real imgs...")

        print("Building gradcam grid for pre-attacked imgs...")

        plot_clean_gradcam_grid_across_models(
            models_data,
            true_class=real_label.item(),
            save_path_clean=f"gradcam/original_samples/model_compare_original_real_{i}.png"
        )

        for attack in attack_names:
            for epsilon in epsilons:

                eps = epsilon * 255
                print(f"Analyzing gradcam for {attack} with epsilon {eps}/255")

                plot_gradcam_grid_across_models(
                    models_data,
                    attack_name=attack,
                    epsilons=[2/255, 4/255, 8/255],
                    eps=epsilon,
                    true_class=real_label.item(),
                    save_path=f"gradcam/model_compare_{attack}_eps{eps}_real_{i}.png"
                )

        # gradcam analysis for the clean image
        #run_gradcam_analysis(
        #    model=model,
        #    normalize=normalize,
        #    gradcam=gradcam,
        #    x_raw=real_raw,
        #    y=real_label,
        #    attacked_raw=attacked_real,
        #    attack_names=attack_names,
        #    epsilons=epsilons,
        #    attacked_save_path=f"gradcam/attacked_real_sample_{model_name}_{i+1}.png",
        #    original_save_path=f"gradcam/original_real_sample_{model_name}_{i+1}.png"
        #)

    fake_list = samples["fake"]
    for i, fake in enumerate(fake_list):
        
        fake_raw = fake["image"].to(device).unsqueeze(0)
        fake_label = fake["label"].to(device).unsqueeze(0)
        attacked_fake = generate_attack_grid(model, normalize, fake_raw, fake_label)

        target_layer = model.layer4[-1].conv3

        gradcam = GradCAM(
            model,
            target_layer
        )

        # gradcam analysis for the clean image
        #run_gradcam_analysis(
        #    model=model,
        #    normalize=normalize,
        #    gradcam=gradcam,
        #    x_raw=fake_raw,
        #    y=fake_label,
        #    attacked_raw=attacked_fake,
        #    attack_names=attack_names,
        #    epsilons=epsilons,
        #    attacked_save_path=f"gradcam/attacked_fake_sample_{model_name}_{i+1}.png",
        #    original_save_path=f"gradcam/original_fake_sample_{model_name}_{i+1}.png"
        #)
        models_data = build_models_data(models_names, normalize, fake_raw, fake_label, gradcam=gradcam)

        print("Building gradcam for fake imgs...")

        plot_clean_gradcam_grid_across_models(
            models_data,
            true_class=fake_label.item(),
            save_path_clean=f"gradcam/original_samples/model_compare_original_fake_{i}.png"
        )

        for attack in attack_names:
            for epsilon in epsilons:

                eps = epsilon * 255
                print(f"Analyzing gradcam for {attack} with epsilon {eps}/255")

                plot_gradcam_grid_across_models(
                    models_data,
                    attack_name=attack,
                    epsilons=[2/255, 4/255, 8/255],
                    eps=epsilon,
                    true_class=fake_label.item(),
                    save_path=f"gradcam/model_compare_{attack}_eps{eps}_fake_{i}.png"
                )


##############################

def plot_gradcam_models_comparison(
    clean_raw,
    clean_cams,
    clean_logits,
    attacked_raw,
    attacked_cams,
    attacked_logits,
    model_names,
    true_class,
    attack_name,
    epsilons,
    save_path=None
):
    """
    Compare Grad-CAM visualizations across multiple models and
    multiple adversarial perturbation strengths.

    Grid structure:
        - Columns: models
        - Row 0: clean image
        - Rows 1...N: adversarial images for each epsilon

    Parameters
    ----------
    clean_raw : Tensor
        Clean input image.
        Shape: [1, 3, H, W]

    clean_cams : dict
        Grad-CAM maps for each model on the clean image.

        Example:
        {
            "Baseline": Tensor [1, H, W],
            "FGSM-AT": Tensor [1, H, W],
            "PGD-AT": Tensor [1, H, W],
            "ADES": Tensor [1, H, W]
        }

    clean_logits : dict
        Logits produced by each model on the clean image.

        Example:
        {
            "Baseline": Tensor [2],
            "FGSM-AT": Tensor [2],
            ...
        }

    attacked_raw : dict
        Adversarial images for each model and epsilon.

        Example:
        {
            "Baseline": [
                Tensor [1, 3, H, W],  # epsilons[0]
                Tensor [1, 3, H, W],  # epsilons[1]
                Tensor [1, 3, H, W]   # epsilons[2]
            ],
            ...
        }

    attacked_cams : dict
        Grad-CAM maps for each model and epsilon.

        Example:
        {
            "Baseline": [
                Tensor [1, H, W],  # epsilons[0]
                Tensor [1, H, W],  # epsilons[1]
                Tensor [1, H, W]   # epsilons[2]
            ],
            ...
        }

    attacked_logits : dict
        Logits for each model and epsilon.

        Example:
        {
            "Baseline": [
                Tensor [2],  # epsilons[0]
                Tensor [2],  # epsilons[1]
                Tensor [2]   # epsilons[2]
            ],
            ...
        }

    model_names : list
        Names of the models.

    true_class : int or Tensor
        Ground-truth class.

    attack_name : str
        Name of the adversarial attack.

    epsilons : list
        List of epsilon values.

        Example:
        [2/255, 4/255, 8/255]

    save_path : str or Path, optional
        Path where the figure will be saved.
    """

    n_models = len(model_names)
    n_epsilons = len(epsilons)

    # Rows = clean + one row per epsilon
    n_rows = n_epsilons + 1
    n_cols = n_models

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.8 * n_rows)
    )

    # Handle edge cases
    if n_models == 1 and n_rows == 1:
        axes = axes.reshape(1, 1)

    elif n_models == 1:
        axes = axes.reshape(n_rows, 1)

    elif n_rows == 1:
        axes = axes.reshape(1, n_models)

    # Convert clean image once
    clean_img = tensor_to_image(clean_raw[0])

    # =========================================================
    # ROW 0 — CLEAN IMAGE
    # =========================================================

    for col, model_name in enumerate(model_names):

        ax = axes[0, col]

        # Clean Grad-CAM
        cam_np = (
            clean_cams[model_name]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        # Prediction
        logits = clean_logits[model_name]

        probs = torch.softmax(
            logits,
            dim=0
        )

        pred = logits.argmax().item()
        conf = probs[pred].item()

        # Plot image
        ax.imshow(clean_img)

        # Plot Grad-CAM
        ax.imshow(
            cam_np,
            cmap="jet",
            alpha=0.45
        )

        # Column titles only on first row
        ax.set_title(
            f"{model_name}\n"
            f"pred={pred}, conf={conf:.3f}",
            fontsize=13
        )

        ax.axis("off")

    # Clean row label
    axes[0, 0].text(
        -0.12,
        0.5,
        "Clean",
        ha="right",
        va="center",
        fontsize=15,
        rotation=90,
        transform=axes[0, 0].transAxes
    )

    # =========================================================
    # ADVERSARIAL ROWS — ONE ROW PER EPSILON
    # =========================================================

    for eps_idx, epsilon in enumerate(epsilons):

        # +1 because row 0 is the clean image
        row = eps_idx + 1

        for col, model_name in enumerate(model_names):

            ax = axes[row, col]

            # Adversarial image for this epsilon
            adv_img = tensor_to_image(
                attacked_raw[model_name][eps_idx][0]
            )

            # Grad-CAM for this epsilon
            cam_np = (
                attacked_cams[model_name][eps_idx]
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )

            # Logits for this epsilon
            logits = attacked_logits[model_name][eps_idx]

            probs = torch.softmax(
                logits,
                dim=0
            )

            pred = logits.argmax().item()
            conf = probs[pred].item()

            # Plot image
            ax.imshow(adv_img)

            # Plot Grad-CAM
            ax.imshow(
                cam_np,
                cmap="jet",
                alpha=0.45
            )

            ax.set_title(
                f"pred={pred}, conf={conf:.3f}",
                fontsize=12
            )

            ax.axis("off")

        # Row label
        eps_255 = epsilon * 255

        axes[row, 0].text(
            -0.12,
            0.5,
            f"{attack_name}\n"
            f"ε={eps_255:.0f}/255",
            ha="right",
            va="center",
            fontsize=14,
            rotation=90,
            transform=axes[row, 0].transAxes
        )

    # =========================================================
    # GLOBAL TITLE
    # =========================================================

    fig.suptitle(
        f"Grad-CAM comparison across models — "
        f"true class = {true_class.item()}",
        fontsize=20
    )

    plt.tight_layout()

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"Saved Grad-CAM comparison to {save_path}"
        )

    plt.close(fig)


def generate_attack_sample(
    model,
    normalize,
    x_raw,
    y,
    attack_name,
    eps
):


    # attacks functions
    fgsm = fb.attacks.FGSM()
    ifgsm = fb.attacks.LinfPGD(
            steps=20,
            abs_stepsize=1/255,
            random_start=False)

    # ---------------------------------------------------------
    # Generate all attacks
    # ---------------------------------------------------------

    preprocessing = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        axis=-3  # per PyTorch (C, H, W)
        )
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing, device=device)
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters()),
        input_shape=(3, 224, 224),
        nb_classes=2,
        clip_values=(0.0, 1.0),
        preprocessing=(
            [0.485, 0.456, 0.406],  # mean ImageNet
            [0.229, 0.224, 0.225]   # std ImageNet
        ),
        device_type=device
        )

    # FGSM
    if attack_name == "FGSM":
        _, adv, _ = fgsm(fmodel, x_raw, y, epsilons=eps)
    
    # IFGSM
    elif attack_name == "IFGSM":
        _, adv, _ = ifgsm(fmodel, x_raw, y, epsilons=eps)
        
    # PGD
    elif attack_name == "PGD":
        adv = standard_pgd_attack(model, x_raw.detach(), 
                                       y, eps, alpha=2/255, 
                                       steps=20, normalize=normalize)
        
    elif attack_name == "SQUARE":
        attack_fn = SquareAttack(
                    estimator=classifier,
                    norm=np.inf,     
                    eps=eps,
                    max_iter=5000,
                    p_init=0.8,
                    nb_restarts=1,
                    batch_size=32
                    )
        adv = attack_fn.generate(x=x_raw.cpu().numpy(), y=y.cpu().numpy())
        # ART returns numpy, need to convert imgs_square to tensor to pass it to the model
        adv = torch.from_numpy(adv).float().to(device)
        

    elif attack_name == "NES":
        adv = nes_attack_first_v(model, x_raw, y, eps=eps, 
                                          sigma=0.001, n_samples=50, step_size=2/255, 
                                          n_iters=100, device=device)

    elif attack_name == "GEN":
        attack_fn = GenAttack(
                    model=model, device=device, eps=eps,
                    population_size=6, max_queries=5000,
                    reduced_dim=56,          # 4x reduction on each spatial dim, matches paper's ratio choice
                    upsample_mode="nearest", # paper's default
                )
        adv, success, n_queries = attack_fn.attack_batch(x_raw,y)
        
    return adv.detach()

def gradcam_analysis(models_names, img_raw, true_label, normalize,  attack, epsilons_list, save_path, device):

    attacked_raw = {}
    attacked_cams = {}
    attacked_logits = {}

    clean_cams = {}
    clean_logits = {}
    model_labels = []

    for model_item in models_names:
            model_label = model_item["model_label"]
            model_labels.append(model_label)
            model_name = model_item["model_name"]
            attacked_raw[model_label] = []
            attacked_cams[model_label] = []
            attacked_logits[model_label] = []
    
            for epsilon in epsilons_list:
    
                checkpoint_path = f"{MODELS_DIR}/{model_name}"
                model = reset_model(checkpoint_path,device)
                target_layer = model.layer4[-1].conv3
                model.eval()
                            
                gradcam = GradCAM(
                    model,
                    target_layer
                )
        
                # CLEAN
                x = normalize(img_raw)
                clean_cam, logits = gradcam(
                    x,
                    target_class=true_label
                )
        
                clean_logits[model_label] = logits.squeeze(0)
                clean_cams[model_label] = clean_cam
        
                # ADVERSARIAL 
        
                # I just create the adversarial example for the specific attack and epsilon
                attacked_real = generate_attack_sample(model, normalize, img_raw, 
                                                       true_label, attack_name=attack, eps=epsilon)
        
                attacked_raw[model_label].append(attacked_real)
        
                # adversarial grad-cam
        
                adv = normalize(attacked_real)
                adv_cam, adv_logits = gradcam(
                            adv,
                            target_class=true_label
                        )
        
                attacked_cams[model_label].append(adv_cam)
                attacked_logits[model_label].append(adv_logits.squeeze(0))

    plot_gradcam_models_comparison(
        img_raw, clean_cams, clean_logits,
        attacked_raw, attacked_cams, attacked_logits, model_names=model_labels,
        true_class=true_label, attack_name=attack, epsilons=epsilons_list, save_path=save_path
    )

    
# ADES ambda 50
# PGD-A LINEAR SCHEDULER
# base model
if __name__ == "__main__":
    model_name_list = [{"model_name":"resnet50_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25_freeze.pt", "mode":"baseline", "lambda_mean":0, "loss_type":"", "num_epochs":25},
                       {"model_name":"resnet50_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_50__lr_0.001_seed_42_epochs_25_freeze_norampup.pt", "mode":"ades", "lambda_mean":50, "loss_type":"MAXLOSS_LINEAR_TARGET", "num_epochs":25},
                       {"model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt", "mode":"clean", "lambda_mean":0, "loss_type":"", "num_epochs":12}]

    #model_name_list = [{"model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt", "mode":"clean", "lambda_mean":0, "loss_type":"", "num_epochs":12}]
           
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)
    #model = model.cuda()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  
    models_names = [{"model_label":"baseline model", "model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"},
                            {"model_label":"PGD-AT linear eps sched", "model_name":"pgdat_ades_difat/resnet50_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25_freeze.pt"},
                            {"model_label":"ADES", "model_name":"pgdat_ades_difat/resnet50_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_50__lr_0.001_seed_42_epochs_25_freeze_norampup.pt"},
                            {"model_label":"ADES + 11 epochs PGD-AT", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_ades_baseline_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"PGD-AT + DiFat conf 2", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_7__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"PGD-AT + DiFat conf 4", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_[3, 6, 13]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"},
                            {"model_label":"ADES + PGD-AT + DiFat", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_ades_baseline_difat_epochsbaseline_[14, 15, 16, 17, 18, 19, 20, 21, 22, 23]_epochsdifat_[24]__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"}]

    #models_names = [{"model_label":"PGD-AT + DiFat conf 2", "model_name":"pgdat_mixed_trainings/resnet50_pgdat_baseline_difat_epochsdifat_7__lr_0.001_seed_42_epochs_25_numeprampup_12.pt"}]
    print("Searching for the images....")
    #save_gradcam_samples(
    #    test_loader,
    #    "gradcam_samples.pt",
    #    n_per_class=50
    #)
    #real_index = select_sample_indices(
    #    test_dataset, 1, "real"
    #)
    #fake_indeces = select_sample_indices(
    #    test_dataset, 2, "fake"
    #)

    samples = torch.load(
        "gradcam_samples.pt",
        map_location="cpu"
    )

    # "r.c23.{vid1}.{frame.split('.')[0]}"
    # "f.c23.{vid1}_{vid2}.{frame.split('.')[0]}"
    # real "r.c23.784.00082"
    # fake "f.c23.784_769.0082"

    #real_ids = [k for k in test_dataset.id_to_idx if k.startswith("r.c23.")]
    #print(len(real_ids))
    #print([k for k in real_ids if "949" in k][:10])

    #real_raw, real_label, img_id = test_dataset.get_by_id('r.c23.949.frame_00113')
    #fake_raw, fake_label, img_id = test_dataset.get_by_id('f.c23.949_868.frame_00113')

    i = 30
    real_list = samples["real"]
    real = real_list[i]
    fake_list = samples["fake"]
    fake = fake_list[i]

    real_raw = real["image"].to(device).unsqueeze(0)
    real_label = real["label"].to(device).unsqueeze(0)
    fake_raw = fake["image"].to(device).unsqueeze(0)
    fake_label = fake["label"].to(device).unsqueeze(0)

    attacks_list = ["FGSM", "IFGSM", "PGD", "SQUARE", "GEN", "NES"]
    epsilons_list = [2/255, 4/255, 8/255]
    black_box_epsilons_list = [8/255]

    attack = 'NES'

    # REAL image
    save_path = f"gradcam/models_comparison/target_layer_layer4_conv3/gradcam_comparison_{attack}_real_{i}.png"
    gradcam_analysis(models_names, real_raw, real_label, normalize, attack, black_box_epsilons_list, save_path, device)

    # FAKE image
    save_path = f"gradcam/models_comparison/target_layer_layer4_conv3/gradcam_comparison_{attack}_fake_{i}.png"
    gradcam_analysis(models_names, fake_raw, fake_label, normalize, attack, black_box_epsilons_list, save_path, device)
    


        





        

    


    

    