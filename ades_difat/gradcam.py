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
import itertools
import foolbox as fb
from test_attacks import test_attack
from training_ades_difat import standard_pgd_attack
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

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted activation maps
        cam = (weights * activations).sum(
            dim=1,
            keepdim=True
        )

        # ReLU
        cam = F.relu(cam)

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
                                       y, epsilon, alpha=2/255, 
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

# ADES lambda 50
# PGD-AT LINEAR SCHEDULER
# base model
if __name__ == "__main__":
    model_name_list = [{"model_name":"resnet50_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25_freeze.pt", "mode":"baseline", "lambda_mean":0, "loss_type":"", "num_epochs":25},
                       {"model_name":"resnet50_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_50__lr_0.001_seed_42_epochs_25_freeze_norampup.pt", "mode":"ades", "lambda_mean":50, "loss_type":"MAXLOSS_LINEAR_TARGET", "num_epochs":25},
                       {"model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt", "mode":"clean", "lambda_mean":0, "loss_type":"", "num_epochs":12}]

    #model_name_list = [{"model_name":"resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt", "mode":"clean", "lambda_mean":0, "loss_type":"", "num_epochs":12}]
           
        
    for values in model_name_list:
        model_name = values["model_name"]
        mode = values["mode"]
        num_epochs = values["num_epochs"]
        if mode == "ades":
            lambda_mean = values["lambda_mean"]
            loss_type = values["loss_type"]
        results = []
        
        lr = 1e-03
        weight_decay =5e-4
        batch_size = 32
        alpha_adv = 0.5
        sched_type = "linear"
        epsilon = 8/255
        
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
        
        
        checkpoint_path = f"{MODELS_DIR}/pgdat_ades_difat/{model_name}"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  

        print("Searching for the images....")
        #save_gradcam_samples(
        #    test_loader,
        #    "gradcam_samples.pt",
        #    n_per_class=2
        #)

        #real_index = select_sample_indices(
        #    test_dataset, 1, "real"
        #)
#
        #fake_indeces = select_sample_indices(
        #    test_dataset, 2, "fake"
        #)

        samples = torch.load(
            "gradcam_samples.pt",
            map_location="cpu"
        )

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
    
            attack_names = ["FGSM", "IFGSM", "PGD"]
            epsilons = [2/255, 4/255, 8/255]
    
            # gradcam analysis for the clean image
            run_gradcam_analysis(
                model=model,
                normalize=normalize,
                gradcam=gradcam,
                x_raw=real_raw,
                y=real_label,
                attacked_raw=attacked_real,
                attack_names=attack_names,
                epsilons=epsilons,
                attacked_save_path=f"gradcam/attacked_real_sample_{model_name}_{i+1}.png",
                original_save_path=f"gradcam/original_real_sample_{model_name}_{i+1}.png"
            )

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
    
            attack_names = ["FGSM", "IFGSM", "PGD"]
            epsilons = [2/255, 4/255, 8/255]
    
            # gradcam analysis for the clean image
            run_gradcam_analysis(
                model=model,
                normalize=normalize,
                gradcam=gradcam,
                x_raw=fake_raw,
                y=fake_label,
                attacked_raw=attacked_fake,
                attack_names=attack_names,
                epsilons=epsilons,
                attacked_save_path=f"gradcam/attacked_fake_sample_{model_name}_{i+1}.png",
                original_save_path=f"gradcam/original_fake_sample_{model_name}_{i+1}.png"
            )
    
            
    
            