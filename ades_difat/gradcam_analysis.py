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
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from training_ades_difat import standard_pgd_attack

"""
Grad-CAM comparison: clean vs. adversarial images.

Install first:
    pip install grad-cam

Usage:
    from gradcam_analysis import compare_clean_adv_gradcam

    compare_clean_adv_gradcam(
        model=model,
        imgs_raw=imgs_raw,          # [B, C, H, W] RAW pixel-space, [0,1] -- pre-attack
        imgs_adv_raw=imgs_adv_raw,  # [B, C, H, W] RAW pixel-space, [0,1] -- post-attack
        y=y,                        # [B] true labels
        normalize=normalize,        # your existing torchvision Normalize transform
        target_layer=model.layer4[-1],  # last conv block, standard choice for ResNet50
        save_path="gradcam_comparison.png",
        max_samples=8,
    )

Why compare pre/post attack CAMs
---------------------------------
If the model has genuinely learned robust features, Grad-CAM should highlight
similar regions of the face before and after attack -- the adversarial
perturbation shouldn't meaningfully change WHERE the model is looking, even if
it changes the final prediction. If, instead, the attack causes attention to
jump to seemingly irrelevant regions (background, image borders, compression
artifacts) or scatter into a diffuse, incoherent pattern, that's a visual
signature consistent with gradient masking / a brittle, non-robust decision
boundary -- useful supporting evidence alongside your quantitative AUC/ASR
numbers for the catastrophic-overfitting discussion in your thesis.
"""


def compute_gradcam(model, imgs_norm: torch.Tensor, target_layer, targets=None):
    """
    Computes Grad-CAM heatmaps for a batch of ALREADY-NORMALIZED images.

    Args:
        model: your classifier (must be in eval mode for stable CAMs -- Grad-CAM
               itself needs gradients, so don't wrap the call in torch.no_grad()).
        imgs_norm: [B, C, H, W] normalized images (the same tensor you'd pass
               directly to model(...) for a forward pass).
        target_layer: the conv layer to hook into. For ResNet50, `model.layer4[-1]`
               (the last block of the last stage) is the standard choice --
               it's the last spatial feature map before global pooling, so its
               gradients best reflect "where in the image mattered".
        targets: list of `ClassifierOutputTarget(class_idx)`, one per sample,
               or None to use the model's own top-1 prediction per sample
               (i.e. explain whatever the model actually predicted, which is
               usually what you want for a pre/post-attack comparison -- you're
               explaining the DECISION, not forcing an explanation for the
               true label).

    Returns:
        heatmaps: [B, H, W] numpy array, values in [0, 1].
    """
    cam = GradCAM(model=model, target_layers=[target_layer])
    heatmaps = cam(input_tensor=imgs_norm, targets=targets)  # [B, H, W]
    return heatmaps


def _denorm_for_display(img_raw_single: torch.Tensor) -> np.ndarray:
    """
    img_raw_single: [C, H, W] tensor in RAW [0,1] pixel space (NOT normalized).
    Returns an [H, W, C] float32 numpy array in [0,1], as required by
    show_cam_on_image.
    """
    img = img_raw_single.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1).astype(np.float32)


def compare_clean_adv_gradcam(model, imgs_raw: torch.Tensor, imgs_adv_raw: torch.Tensor,
                               y: torch.Tensor, normalize, target_layer,
                               save_path: str = "gradcam_comparison.png",
                               max_samples: int = 8, use_predicted_class: bool = True):
    """
    Runs Grad-CAM on a batch of clean images and their adversarial counterparts,
    and saves a grid: for each sample, [clean image+CAM | adv image+CAM].

    Args:
        imgs_raw / imgs_adv_raw: RAW [0,1] pixel-space images, pre/post attack,
               SAME order/identity per sample (i.e. imgs_adv_raw[i] must be the
               attacked version of imgs_raw[i]).
        y: [B] true labels (used only for the printed title/labels, not for
               choosing the CAM target when use_predicted_class=True).
        normalize: your torchvision Normalize transform.
        target_layer: e.g. model.layer4[-1] for ResNet50.
        use_predicted_class: if True, each image's CAM explains the model's
               OWN top-1 prediction on that specific image (clean CAM explains
               the clean prediction, adv CAM explains the adv prediction --
               this is what you want to see "why did it flip"). If False,
               both CAMs explain the true label instead (useful for a
               different question: "does attention on the correct class
               degrade under attack", even before checking if the prediction
               flipped).
    """
    model.eval()
    n = min(max_samples, imgs_raw.size(0))
    imgs_raw = imgs_raw[:n]
    imgs_adv_raw = imgs_adv_raw[:n]
    y = y[:n]

    imgs_norm = normalize(imgs_raw)
    imgs_adv_norm = normalize(imgs_adv_raw)

    if use_predicted_class:
        with torch.no_grad():
            pred_clean = model(imgs_norm).argmax(dim=1)
            pred_adv = model(imgs_adv_norm).argmax(dim=1)
        targets_clean = [ClassifierOutputTarget(int(p)) for p in pred_clean]
        targets_adv = [ClassifierOutputTarget(int(p)) for p in pred_adv]
    else:
        targets_clean = [ClassifierOutputTarget(int(label)) for label in y]
        targets_adv = targets_clean

    cams_clean = compute_gradcam(model, imgs_norm, target_layer, targets=targets_clean)
    cams_adv = compute_gradcam(model, imgs_adv_norm, target_layer, targets=targets_adv)

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i in range(n):
        clean_img_np = _denorm_for_display(imgs_raw[i])
        adv_img_np = _denorm_for_display(imgs_adv_raw[i])

        clean_overlay = show_cam_on_image(clean_img_np, cams_clean[i], use_rgb=True)
        adv_overlay = show_cam_on_image(adv_img_np, cams_adv[i], use_rgb=True)

        pred_c = int(pred_clean[i]) if use_predicted_class else int(y[i])
        pred_a = int(pred_adv[i]) if use_predicted_class else int(y[i])

        axes[i, 0].imshow(clean_overlay)
        axes[i, 0].set_title(f"Clean | true={int(y[i])} pred={pred_c}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(adv_overlay)
        axes[i, 1].set_title(f"Adversarial | true={int(y[i])} pred={pred_a}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM comparison to {save_path}")

EPS_LIST = [2 / 255, 4 / 255, 8 / 255]
 
 
def _select_one_real_one_fake(imgs_raw: torch.Tensor, y: torch.Tensor):
    """Picks the first sample with y==0 (real) and the first with y==1 (fake)."""
    idx_real = (y == 0).nonzero(as_tuple=True)[0]
    idx_fake = (y == 1).nonzero(as_tuple=True)[0]
    assert len(idx_real) > 0, "no real (y=0) sample found in this batch"
    assert len(idx_fake) > 0, "no fake (y=1) sample found in this batch"
    return idx_real[0].item(), idx_fake[0].item()
 
 
def _generate_attacked_versions(fmodel, normalize, img_raw: torch.Tensor, label: torch.Tensor):
    """
    img_raw: [1, C, H, W] RAW [0,1], single sample.
    label:   [1] label for that sample.
 
    Returns a dict: {"FGSM eps=2/255": adv_tensor, ...} for all 9 combinations,
    each adv_tensor shaped [1, C, H, W] in RAW [0,1] space.
    """
    fgsm = fb.attacks.FGSM()
    ifgsm = fb.attacks.LinfPGD(
            steps=20,
            abs_stepsize=1/255,
            random_start=False)  # → with no random start PGD = IFGSM
 
    results = {}

    for eps in EPS_LIST:
        # FGSM
        _, adv, _ = fgsm(fmodel, img_raw, label, epsilons=eps)
        eps_255 = round(eps * 255)
        results[f"FGSM eps={eps_255}/255"] = adv
        # IFGSM
        _, adv, _ = ifgsm(fmodel, img_raw, label, epsilons=eps)
        results[f"IFGSM eps={eps_255}/255"] = adv
        # PGD
        adv = standard_pgd_attack(model, img_raw.detach(), label, eps, 2/255, 20, normalize)
        results[f"PGD eps={eps_255}/255"] = adv
 
    return results
 
 
def run_attack_gradcam_grid(model, fmodel, imgs_raw: torch.Tensor, y: torch.Tensor,
                             normalize, target_layer,
                             save_path: str = "gradcam_attack_grid.png",
                             use_predicted_class: bool = True):
    model.eval()
    idx_real, idx_fake = _select_one_real_one_fake(imgs_raw, y)
 
    rows = []  # each row: (row_label, clean_img_raw[1,C,H,W], label[1], {name: adv_img_raw})
    for idx, row_label in [(idx_real, "REAL sample"), (idx_fake, "FAKE sample")]:
        img_raw = imgs_raw[idx:idx + 1]
        label = y[idx:idx + 1]
        attacked = _generate_attacked_versions(fmodel, normalize, img_raw, label)
        rows.append((row_label, img_raw, label, attacked))
 
    col_names = ["Clean"] + list(rows[0][3].keys())  # same 9 attack/eps names for both rows
    n_cols = len(col_names)
 
    fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 5))
 
    for r, (row_label, img_raw, label, attacked) in enumerate(rows):
        # build the ordered list of images for this row: clean, then 9 attacked
        images_this_row = [("Clean", img_raw)] + list(attacked.items())
 
        for c, (col_name, img_variant_raw) in enumerate(images_this_row):
            img_norm = normalize(img_variant_raw)
 
            if use_predicted_class:
                with torch.no_grad():
                    pred = model(img_norm).argmax(dim=1)
                targets = [ClassifierOutputTarget(int(pred[0]))]
            else:
                targets = [ClassifierOutputTarget(int(label[0]))]
 
            cam = compute_gradcam(model, img_norm, target_layer, targets=targets)
            img_np = _denorm_for_display(img_variant_raw[0])
            overlay = show_cam_on_image(img_np, cam[0], use_rgb=True)
 
            ax = axes[r, c]
            ax.imshow(overlay)
            pred_label = int(pred[0]) if use_predicted_class else int(label[0])
            title = col_name if c == 0 else f"{col_name}\npred={pred_label}"
            if c == 0:
                title = f"{col_name}\npred={pred_label}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")
 
        axes[r, 0].set_ylabel(row_label, fontsize=10)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM attack grid to {save_path}")



if __name__ == "__main__":

    model_name_list = [{"model_name":"resnet50_pgdat_ades_MAXLOSS_LINEAR_TARGET_lambda_mean_8.0__lr_0.001_seed_42_epochs_15_freeze.pt", "mode":"ades", "lambda_mean":8, "loss_type":"MAXLOSS_LINEAR_TARGET", "num_epochs":15},
                       ]
    
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
        
        test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
        #with open("sampled_test_set.json") as f:
        #    sampled_test_set_paths = json.loads(f.read())
        #test_small = get_imgs_by_filepath(test_dataset, sampled_test_set_paths)
        test_small, _ = balanced_subset(test_dataset, n_per_class=500)
        test_loader = DataLoader(test_small, batch_size=batch_size, shuffle=False)    
        imgs_raw, y, _ = next(iter(test_loader))
        imgs_raw, y = imgs_raw.to(device), y.to(device).long().view(-1)
    
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocessing = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        axis=-3  # per PyTorch (C, H, W)
        )
        fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing, device=device)

        run_attack_gradcam_grid(
            model=model,
            fmodel=fmodel,              # reuse your existing fb.PyTorchModel wrapper
            imgs_raw=imgs_raw,
            y=y,
            normalize=normalize,
            target_layer=model.layer4[-1],
            save_path="gradcam/gradcam_whitebox_grid.png",
        )