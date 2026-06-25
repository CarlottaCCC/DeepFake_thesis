"""
AutoZOOM Attack - PyTorch Port
Original paper: "AutoZOOM: Autoencoder-based Zeroth Order Optimization Method
for Attacking Black-box Neural Networks" (AAAI 2019)
Original TF1 code: https://github.com/IBM/Autozoom-Attack
License: Apache 2.0
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Callable


# ---------------------------------------------------------------------------
# Coordinate-wise ADAM optimizer (numpy, mirrors the original implementation)
# ---------------------------------------------------------------------------

def coordinate_ADAM(losses, indices, grad, hess, mt_arr, vt_arr,
                    real_modifier, lr, adam_epoch, beta1=0.9, beta2=0.999,
                    proj=True, img_min=0.0, img_max=1.0):
    """
    Coordinate-wise ADAM update used by ZOO / AutoZOOM.
    Operates in-place on real_modifier (numpy array).
    """
    for i, idx in enumerate(indices):
        mt_arr[idx] = beta1 * mt_arr[idx] + (1 - beta1) * grad[i]
        vt_arr[idx] = beta2 * vt_arr[idx] + (1 - beta2) * (grad[i] ** 2)
        mt_hat = mt_arr[idx] / (1 - beta1 ** adam_epoch)
        vt_hat = vt_arr[idx] / (1 - beta2 ** adam_epoch)
        delta = lr * mt_hat / (np.sqrt(vt_hat) + 1e-8)
        real_modifier.flat[idx] -= delta

    if proj:
        np.clip(real_modifier, img_min - 0.5, img_max - 0.5, out=real_modifier)


# ---------------------------------------------------------------------------
# C&W-style loss (binary version for deepfake detectors)
# ---------------------------------------------------------------------------

def cw_loss_binary(logits: np.ndarray, target_class: int, confidence: float = 0.0):
    """
    C&W loss for a binary classifier (2 logits: [real, fake]).
    For untargeted attack on class `target_class`, we want to LEAVE that class.
    """
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]  # (1, 2)

    other = 1 - target_class
    loss = np.maximum(0.0, logits[:, target_class] - logits[:, other] + confidence)
    return loss.sum()


# ---------------------------------------------------------------------------
# AutoZOOM Attack class
# ---------------------------------------------------------------------------

class AutoZOOMAttack:
    """
    AutoZOOM black-box adversarial attack in PyTorch.

    Supports two attack modes:
      - 'bilin'  : bilinear downsampling for gradient estimation (autozoom_bilin)
      - 'ae'     : autoencoder-based compression (autozoom_ae)

    The model is treated as a pure black-box: only forward passes are used.

    Args:
        model_fn        : callable, takes a numpy array (N,C,H,W) in [0,1],
                          returns numpy logits (N, num_classes)
        attack_mode     : 'bilin' or 'ae'
        img_shape       : (C, H, W) of the input images
        targeted        : if True, performs a targeted attack
        confidence      : C&W confidence margin
        init_const      : initial regularization constant
        lr              : ADAM learning rate
        max_iterations  : max iterations per image
        switch_iter     : how often to update the regularization constant
        num_rand_vec    : number of random vectors for gradient estimation
        img_resize      : reduced spatial size for 'bilin' mode (e.g. 32)
        encoder_fn      : callable(x) -> z  (only for mode='ae')
        decoder_fn      : callable(z) -> x  (only for mode='ae')
        img_min/img_max : pixel value range
        verbose         : print progress
    """

    def __init__(
        self,
        model_fn: Callable,
        attack_mode: str = 'bilin',
        img_shape: tuple = (3, 224, 224),
        targeted: bool = False,
        confidence: float = 0.0,
        init_const: float = 10.0,
        lr: float = 1e-2,
        max_iterations: int = 10000,
        switch_iter: int = 1000,
        num_rand_vec: int = 1,
        img_resize: int = 32,
        encoder_fn: Optional[Callable] = None,
        decoder_fn: Optional[Callable] = None,
        img_min: float = 0.0,
        img_max: float = 1.0,
        verbose: bool = True,
    ):
        assert attack_mode in ('bilin', 'ae'), "attack_mode must be 'bilin' or 'ae'"
        if attack_mode == 'ae':
            assert encoder_fn is not None and decoder_fn is not None, \
                "encoder_fn and decoder_fn must be provided for mode='ae'"

        self.model_fn = model_fn
        self.attack_mode = attack_mode
        self.img_shape = img_shape          # (C, H, W)
        self.targeted = targeted
        self.confidence = confidence
        self.init_const = init_const
        self.lr = lr
        self.max_iterations = max_iterations
        self.switch_iter = switch_iter
        self.num_rand_vec = num_rand_vec
        self.img_resize = img_resize
        self.encoder_fn = encoder_fn
        self.decoder_fn = decoder_fn
        self.img_min = img_min
        self.img_max = img_max
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_modifier(self, modifier: np.ndarray, orig: np.ndarray) -> np.ndarray:
        """
        Map the (compressed) modifier back to image space and add to original.
        Returns the adversarial image clipped to [img_min, img_max], shape (C,H,W).
        """
        C, H, W = self.img_shape

        if self.attack_mode == 'bilin':
            # modifier shape: (1, C, img_resize, img_resize)
            mod_t = torch.tensor(modifier, dtype=torch.float32)
            mod_up = F.interpolate(mod_t, size=(H, W), mode='bilinear',
                                   align_corners=False)
            delta = mod_up.numpy()[0]                 # (C, H, W)

        else:  # 'ae'
            # modifier is in latent space; decoder maps it back to image space
            delta = self.decoder_fn(modifier)         # (C, H, W) expected
            if delta.shape != (C, H, W):
                delta = np.reshape(delta, (C, H, W))

        adv = np.clip(orig + delta, self.img_min, self.img_max)
        return adv

    def _query_model(self, adv_img: np.ndarray, orig_img: np.ndarray,
                     modifier: np.ndarray, target_class: int,
                     const: float) -> tuple:
        """
        Query the model and compute the C&W total loss.
        Returns (total_loss, attack_loss, l2_dist, logits).
        """
        adv_batch = adv_img[np.newaxis]              # (1, C, H, W)
        logits = self.model_fn(adv_batch)             # (1, num_classes)

        attack_loss = cw_loss_binary(logits, target_class, self.confidence)

        delta = adv_img - orig_img
        l2 = float(np.sum(delta ** 2))

        if self.targeted:
            total_loss = l2 + const * attack_loss
        else:
            # For untargeted: we MAXIMISE attack_loss, so minimise -attack_loss
            total_loss = l2 - const * attack_loss

        return total_loss, attack_loss, l2, logits

    def _estimate_gradient(self, orig_img: np.ndarray, modifier: np.ndarray,
                           target_class: int, const: float, h: float = 0.0001):
        """
        Zeroth-order finite-difference gradient estimation.
        Uses `num_rand_vec` random directions in the compressed space.
        """
        grad = np.zeros_like(modifier)
        flat_mod = modifier.flatten()
        n = len(flat_mod)

        for _ in range(self.num_rand_vec):
            # Random unit vector in compressed space
            u = np.random.randn(n).astype(np.float32)
            u /= (np.linalg.norm(u) + 1e-8)

            mod_plus = (flat_mod + h * u).reshape(modifier.shape)
            mod_minus = (flat_mod - h * u).reshape(modifier.shape)

            adv_plus = self._decode_modifier(mod_plus, orig_img)
            adv_minus = self._decode_modifier(mod_minus, orig_img)

            loss_plus, _, _, _ = self._query_model(adv_plus, orig_img, mod_plus,
                                                   target_class, const)
            loss_minus, _, _, _ = self._query_model(adv_minus, orig_img, mod_minus,
                                                    target_class, const)

            grad_est = (loss_plus - loss_minus) / (2 * h) * u
            grad += grad_est.reshape(modifier.shape)

        return grad / self.num_rand_vec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attack(self, image: np.ndarray, true_label: int,
               target_label: Optional[int] = None) -> np.ndarray:
        """
        Generate an adversarial example for a single image.

        Args:
            image       : numpy array, shape (C, H, W), range [img_min, img_max]
            true_label  : ground-truth class index
            target_label: target class for targeted attacks (ignored if untargeted)

        Returns:
            best_adv    : adversarial image, shape (C, H, W)
        """
        C, H, W = self.img_shape
        target_class = target_label if (self.targeted and target_label is not None) \
                       else true_label

        # Initialise modifier in compressed space
        if self.attack_mode == 'bilin':
            mod_shape = (1, C, self.img_resize, self.img_resize)
        else:
            latent = self.encoder_fn(image[np.newaxis])   # (1, latent_dim) or similar
            mod_shape = latent.shape

        modifier = np.zeros(mod_shape, dtype=np.float32)
        mt_arr   = np.zeros(modifier.size, dtype=np.float32)
        vt_arr   = np.zeros(modifier.size, dtype=np.float32)

        const = self.init_const
        best_l2 = np.inf
        best_adv = image.copy()
        adam_epoch = 1

        for iteration in range(1, self.max_iterations + 1):

            # Decode modifier → adversarial image
            adv_img = self._decode_modifier(modifier, image)

            # Evaluate loss
            total_loss, attack_loss, l2, logits = self._query_model(
                adv_img, image, modifier, target_class, const
            )

            # Update best result
            success = (attack_loss <= 0.0) if not self.targeted else (attack_loss <= 0.0)
            if success and l2 < best_l2:
                best_l2 = l2
                best_adv = adv_img.copy()

            # Estimate gradient in compressed space
            grad = self._estimate_gradient(image, modifier, target_class, const)

            # Coordinate-wise ADAM step (all coordinates at once)
            indices = np.arange(modifier.size)
            coordinate_ADAM(
                losses=None,
                indices=indices,
                grad=grad.flatten(),
                hess=None,
                mt_arr=mt_arr,
                vt_arr=vt_arr,
                real_modifier=modifier,
                lr=self.lr,
                adam_epoch=adam_epoch,
                proj=True,
                img_min=self.img_min,
                img_max=self.img_max,
            )
            adam_epoch += 1

            # Update regularization constant
            if iteration % self.switch_iter == 0:
                if attack_loss > 0:
                    const *= 10     # attack failing: increase pressure
                else:
                    const /= 10     # attack succeeding: reduce distortion
                const = np.clip(const, 1e-4, 1e4)

            if self.verbose and iteration % 500 == 0:
                pred = int(np.argmax(logits[0]))
                print(f"[AutoZOOM] iter={iteration:5d} | loss={total_loss:.4f} | "
                      f"l2={l2:.4f} | attack_loss={attack_loss:.4f} | "
                      f"pred={pred} | const={const:.4f}")

        return best_adv

    def generate(self, images: np.ndarray, labels: np.ndarray,
                 target_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Batch entry point. Attacks each image independently.

        Args:
            images        : (N, C, H, W)
            labels        : (N,) true labels
            target_labels : (N,) target labels (optional, for targeted mode)

        Returns:
            adv_images    : (N, C, H, W)
        """
        adv_images = images.copy()
        for i in range(len(images)):
            tgt = int(target_labels[i]) if target_labels is not None else None
            if self.verbose:
                print(f"\n[AutoZOOM] Attacking image {i+1}/{len(images)} "
                      f"(label={labels[i]})")
            adv_images[i] = self.attack(images[i], int(labels[i]), tgt)
        return adv_images