"""
AutoZOOM - autozoom_bilin variant
Paper: "AutoZOOM: Autoencoder-based Zeroth Order Optimization Method for
Attacking Black-box Neural Networks" (AAAI 2019)

Questa implementazione usa il downsampling bilineare come spazio ridotto
(senza autoencoder), compatibile con il pattern ART BlackBoxAttack.

Dipendenze: torch, numpy, art
"""

import numpy as np
import torch
import torch.nn.functional as F
from art.attacks.attack import EvasionAttack
from art.estimators.classification import PyTorchClassifier
from typing import Optional


class AutoZoomBilin(EvasionAttack):
    """
    Variante autozoom_bilin di AutoZOOM.

    La stima del gradiente avviene nello spazio ridotto (downsampling bilineare),
    poi viene riproiettata alla risoluzione originale prima dell'update Adam.
    Questo riduce drasticamente il numero di query rispetto a ZOO coordinata-per-coordinata.

    Args:
        estimator:   PyTorchClassifier wrappato con ART.
        max_iter:    Numero massimo di iterazioni Adam.
        learning_rate: Learning rate Adam.
        binary_search_steps: Passi di binary search sulla costante c (loss trade-off).
        init_const:  Valore iniziale della costante c.
        confidence:  Margine kappa sulla loss C&W.
        targeted:    Se True, attacco mirato (y deve essere la classe target).
        reduce_factor: Fattore di riduzione bilineare (es. 4 → immagine /4 per lato).
        num_random_vecs: Vettori casuali per stima RGE per iterazione.
        h:           Passo di differenza finita per RGE.
        clip_min/max: Clipping pixel.
        verbose:     Stampa progress ogni N iterazioni (0 = silenzioso).
    """

    attack_params = EvasionAttack.attack_params + [
        "max_iter",
        "learning_rate",
        "binary_search_steps",
        "init_const",
        "confidence",
        "targeted",
        "reduce_factor",
        "num_random_vecs",
        "h",
        "clip_min",
        "clip_max",
        "verbose",
    ]

    _estimator_requirements = (PyTorchClassifier,)

    def __init__(
        self,
        estimator: PyTorchClassifier,
        max_iter: int = 1000,
        learning_rate: float = 1e-2,
        binary_search_steps: int = 5,
        init_const: float = 1.0,
        confidence: float = 0.0,
        targeted: bool = False,
        reduce_factor: int = 4,
        num_random_vecs: int = 1,
        h: float = 1e-4,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        verbose: int = 100,
    ):
        super().__init__(estimator=estimator)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.init_const = init_const
        self.confidence = confidence
        self.targeted = targeted
        self.reduce_factor = reduce_factor
        self.num_random_vecs = num_random_vecs
        self.h = h
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.verbose = verbose
        self._check_params()

    def _check_params(self) -> None:
        if self.reduce_factor < 1:
            raise ValueError("reduce_factor deve essere >= 1")
        if self.num_random_vecs < 1:
            raise ValueError("num_random_vecs deve essere >= 1")

    # ------------------------------------------------------------------
    # Interfaccia ART
    # ------------------------------------------------------------------

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Genera esempi adversarial.

        Args:
            x: Input originali, shape (N, C, H, W), range [clip_min, clip_max].
            y: Label originali one-hot o indici. Se targeted, label target.

        Returns:
            x_adv: Esempi adversarial, stessa shape di x.
        """
        if y is None:
            # Usa le predizioni del modello come label originali
            preds = self.estimator.predict(x)
            y = np.argmax(preds, axis=1)
        elif y.ndim > 1:
            y = np.argmax(y, axis=1)

        x_adv = x.copy()
        for i in range(x.shape[0]):
            x_adv[i] = self._attack_single(x[i], y[i])
        return x_adv

    # ------------------------------------------------------------------
    # Core attack su singola immagine
    # ------------------------------------------------------------------

    def _attack_single(self, x: np.ndarray, y: int) -> np.ndarray:
        """Attacca una singola immagine con binary search sulla costante c."""
        const = self.init_const
        best_adv = x.copy()
        best_l2 = float("inf")

        for bs in range(self.binary_search_steps):
            adv, l2, success = self._inner_attack(x, y, const)
            if success and l2 < best_l2:
                best_l2 = l2
                best_adv = adv
                const /= 2.0          # perturbazione trovata → riduci c
            else:
                const *= 10.0         # non trovata → aumenta c

        return best_adv

    def _inner_attack(self, x: np.ndarray, y: int, const: float):
        """
        Loop Adam con stima RGE nello spazio bilineare ridotto.

        Ritorna (x_adv, l2_dist, success).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Forma (C, H, W) → (1, C, H, W)
        C, H, W = x.shape
        x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

        # δ inizializzato a zero nello spazio originale
        delta = torch.zeros_like(x_t, requires_grad=False)

        # Stato Adam
        m = torch.zeros_like(delta)
        v = torch.zeros_like(delta)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        # Dimensioni spazio ridotto
        h_r = max(1, H // self.reduce_factor)
        w_r = max(1, W // self.reduce_factor)

        best_adv = x.copy()
        best_l2 = float("inf")
        success = False

        for t in range(1, self.max_iter + 1):
            x_adv_t = torch.clamp(x_t + delta, self.clip_min, self.clip_max)
            grad_full = self._estimate_gradient_bilin(x_adv_t, x_t, y, const, h_r, w_r, device)

            # Adam update
            m = beta1 * m + (1 - beta1) * grad_full
            v = beta2 * v + (1 - beta2) * grad_full ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            delta = delta - self.learning_rate * m_hat / (v_hat.sqrt() + eps_adam)

            # Proietta δ in modo che x+δ rimanga nel range
            delta = torch.clamp(x_t + delta, self.clip_min, self.clip_max) - x_t

            # Valuta qualità dell'esempio corrente
            x_adv_np = (x_t + delta).squeeze(0).cpu().numpy()
            l2 = float(np.linalg.norm(x_adv_np - x))
            pred = np.argmax(self.estimator.predict(x_adv_np[None]))

            attacked = (pred != y) if not self.targeted else (pred == y)
            if attacked and l2 < best_l2:
                best_l2 = l2
                best_adv = x_adv_np.copy()
                success = True

            if self.verbose > 0 and t % self.verbose == 0:
                print(f"  iter {t:4d}/{self.max_iter}  l2={l2:.4f}  pred={pred}  target={y}  success={success}")

        return best_adv, best_l2, success

    # ------------------------------------------------------------------
    # Stima gradiente RGE con downsampling bilineare
    # ------------------------------------------------------------------

    def _estimate_gradient_bilin(
        self,
        x_adv: torch.Tensor,   # (1, C, H, W)
        x_orig: torch.Tensor,  # (1, C, H, W) — immagine originale non perturbata
        y: int,
        const: float,
        h_r: int,
        w_r: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Stima ∇_δ L tramite Random Gradient Estimation nello spazio ridotto.

        Per ogni vettore casuale u ~ Uniform(sphere):
            g ≈ (d/du) L * u  usando differenza finita forward
        Poi upsample bilineare di g → spazio originale.
        """
        C, H, W = x_adv.shape[1], x_adv.shape[2], x_adv.shape[3]
        grad_accum = torch.zeros_like(x_adv)

        for _ in range(self.num_random_vecs):
            # Vettore casuale nello spazio ridotto, normalizzato
            u = torch.randn(1, C, h_r, w_r, device=device)
            u = u / (u.norm() + 1e-12)

            # Upsample u → spazio originale
            u_full = F.interpolate(u, size=(H, W), mode="bilinear", align_corners=False)

            # Valuta loss nei due punti (black-box: solo forward pass)
            x_plus = torch.clamp(x_adv + self.h * u_full, self.clip_min, self.clip_max)
            x_minus = torch.clamp(x_adv - self.h * u_full, self.clip_min, self.clip_max)

            loss_plus = self._cw_loss(x_plus, x_orig, y, const)
            loss_minus = self._cw_loss(x_minus, x_orig, y, const)

            # Stima direzionale → proietta su u_full
            directional_deriv = (loss_plus - loss_minus) / (2.0 * self.h)
            grad_accum = grad_accum + directional_deriv * u_full

        return grad_accum / self.num_random_vecs

    # ------------------------------------------------------------------
    # Loss C&W (f6)
    # ------------------------------------------------------------------

    def _cw_loss(self, x: torch.Tensor, x_orig: torch.Tensor, y: int, const: float) -> float:
        """
        Loss C&W = ||δ||² + const * f(x+δ)

        f(x) = max( Z[y_true] - max_{j≠y_true} Z[j] + κ, 0 )   (untargeted)
        f(x) = max( max_{j≠y_tgt} Z[j] - Z[y_tgt] + κ, 0 )     (targeted)
        """
        x_np = x.squeeze(0).detach().cpu().numpy()
        logits = self._get_logits(x_np)
        logits_t = torch.tensor(logits, dtype=torch.float32)

        if not self.targeted:
            z_true = logits_t[y]
            mask = torch.ones_like(logits_t, dtype=torch.bool)
            mask[y] = False
            z_other = logits_t[mask].max()
            f_val = torch.clamp(z_true - z_other + self.confidence, min=0.0)
        else:
            z_target = logits_t[y]
            mask = torch.ones_like(logits_t, dtype=torch.bool)
            mask[y] = False
            z_other = logits_t[mask].max()
            f_val = torch.clamp(z_other - z_target + self.confidence, min=0.0)

        # norma della perturbazione reale δ = x_adv - x_orig
        delta = x.squeeze(0) - x_orig.squeeze(0)
        l2 = float(torch.norm(delta).item())
        return float(l2 + const * f_val.item())

    def _get_logits(self, x_np: np.ndarray) -> np.ndarray:
        """Ottieni logit dal modello tramite ART (black-box, solo forward)."""
        # ART predict restituisce probabilità; usiamo predict_step se disponibile
        if hasattr(self.estimator, "predict_step"):
            return self.estimator.predict_step(x_np[None]).squeeze(0)
        return self.estimator.predict(x_np[None]).squeeze(0)