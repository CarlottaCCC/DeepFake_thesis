"""
GenAttack: Practical Black-box Attacks with Gradient-Free Optimization
Alzantot et al., 2019 (GECCO) — https://arxiv.org/abs/1805.11090

Implements Algorithm 1 from the paper, including the two ImageNet-scale
optimizations from Section 4.1:
  1. Dimensionality reduction: the population is searched as a delta in a
     smaller (reduced_dim x reduced_dim) noise space and upsampled to the
     input resolution before being applied/evaluated. Set reduced_dim=None
     to disable this and search directly in pixel space (matches their
     CIFAR-10/MNIST setup, which used no reduction).
  2. Adaptive parameter scaling: rho (mutation probability) and alpha
     (mutation range) decay when the search plateaus (Eqs. 1-2).

Written for a binary classifier (e.g. real/fake deepfake detection), where
"targeted" attack naturally reduces to "attack toward the other class."
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


class GenAttack:
    def __init__(
        self,
        model,
        device,
        eps=8 / 255,
        population_size=6,          # N in the paper (their CIFAR-10 setting)
        mutation_prob=5e-2,         # rho
        mutation_range=1.0,         # alpha
        tau=0.1,                    # softmax temperature for parent selection
        max_generations=None,       # G; if None, derived from max_queries
        max_queries=5000,
        adaptive=True,              # anneal rho/alpha on plateau (paper section 4.1.2)
        rho_min=0.1,
        alpha_min=0.15,
        plateau_patience=100,       # generations with no improvement before decay
        clip_min=0.0,
        clip_max=1.0,
        reduced_dim=None,           # e.g. 56 for 224x224 inputs; None = no reduction
        upsample_mode="nearest",    # paper uses nearest-neighbor upsampling
    ):
        self.model = model
        self.device = device
        self.eps = eps
        self.N = population_size
        self.rho = mutation_prob
        self.alpha = mutation_range
        self.tau = tau
        self.max_queries = max_queries
        self.adaptive = adaptive
        self.rho_min = rho_min
        self.alpha_min = alpha_min
        self.plateau_patience = plateau_patience
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.reduced_dim = reduced_dim
        self.upsample_mode = upsample_mode

        # each generation costs ~N queries (one fitness eval per member)
        self.max_generations = max_generations or (max_queries // population_size)

    # ------------------------------------------------------------------
    # Search-space <-> pixel-space helpers
    # ------------------------------------------------------------------
    def _search_shape(self, x_orig):
        """Shape of the noise tensors we actually search over."""
        c, h, w = x_orig.shape[1:]
        if self.reduced_dim is None:
            return (c, h, w)
        return (c, self.reduced_dim, self.reduced_dim)

    def _upsample(self, delta, x_orig):
        """Upsample a low-dim delta to the input resolution. No-op if
        dimensionality reduction is disabled."""
        if self.reduced_dim is None:
            return delta
        target_hw = x_orig.shape[-2:]
        kwargs = {"mode": self.upsample_mode}
        if self.upsample_mode not in ("nearest", "nearest-exact"):
            kwargs["align_corners"] = False
        return F.interpolate(delta, size=target_hw, **kwargs)

    def _apply_delta(self, delta, x_orig):
        """Upsample delta (if needed), clip to the eps ball, add to x_orig,
        clip to valid pixel range. delta and x_orig are both batched."""
        delta_full = self._upsample(delta, x_orig)
        delta_full = delta_full.clamp(-self.eps, self.eps)
        x_adv = (x_orig + delta_full).clamp(self.clip_min, self.clip_max)
        return x_adv

    # ------------------------------------------------------------------
    # Fitness: ComputeFitness(x) = log f(x)_t - log sum_{j != t} f(x)_j
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_fitness(self, population_delta, x_orig, target):
        """
        population_delta: (N, C, h, w) in search-space resolution
        x_orig: (1, C, H, W) at full input resolution
        target: int, target class index
        returns: (N,) fitness scores, (N, num_classes) log-probs
        """
        x_orig_batch = x_orig.expand(population_delta.shape[0], -1, -1, -1)
        population = self._apply_delta(population_delta, x_orig_batch)

        logits = self.model(population.to(self.device))
        log_probs = F.log_softmax(logits, dim=1)

        target_log_prob = log_probs[:, target]

        mask = torch.ones_like(log_probs, dtype=torch.bool)
        mask[:, target] = False
        other_log_probs = log_probs.masked_fill(~mask, float("-inf"))
        other_log_sum = torch.logsumexp(other_log_probs, dim=1)

        fitness = target_log_prob - other_log_sum
        return fitness, log_probs, population

    def _crossover(self, parent1, parent2, fitness1, fitness2):
        # p = fitness(parent1) / (fitness(parent1) + fitness(parent2))
        f1 = fitness1.clamp(min=1e-8)
        f2 = fitness2.clamp(min=1e-8)
        p = f1 / (f1 + f2)

        mask = (torch.rand_like(parent1) < p)
        child = torch.where(mask, parent1, parent2)
        return child

    def _mutate(self, child_delta):
        """Mutation happens entirely in search space — no need to touch
        x_orig here, unlike the pixel-space version."""
        mutation_mask = (torch.rand_like(child_delta) < self.rho).float()
        noise = (torch.rand_like(child_delta) * 2 - 1) * (self.alpha * self.eps)
        child_delta = child_delta + mutation_mask * noise
        child_delta = child_delta.clamp(-self.eps, self.eps)
        return child_delta

    def _update_params(self, num_plateaus):
        # Eq. 1-2 in the paper
        self.rho = max(self.rho_min, 0.5 * (0.9 ** num_plateaus))
        self.alpha = max(self.alpha_min, 0.4 * (0.9 ** num_plateaus))

    # ------------------------------------------------------------------
    # Main attack loop — single example at a time (as in the paper)
    # ------------------------------------------------------------------
    def attack_single(self, x_orig, true_label, target=None, verbose=False):
        """
        x_orig: (1, C, H, W) or (C, H, W) tensor, pixel range [clip_min, clip_max]
        true_label: int, ground-truth class
        target: int, target class. Defaults to "the other class" for binary
                classification (i.e. flip real<->fake).
        Returns: (x_adv, success: bool, queries_used: int)
        """
        if x_orig.dim() == 3:
            x_orig = x_orig.unsqueeze(0)
        x_orig = x_orig.to(self.device)

        if target is None:
            num_classes = self.model(x_orig).shape[1]
            if num_classes == 2:
                target = 1 - true_label
            else:
                raise ValueError("target must be specified for num_classes > 2")

        search_c, search_h, search_w = self._search_shape(x_orig)

        # --- initial population: delta ~ U(-eps, eps) in search space ---
        population_delta = (
            torch.rand(self.N, search_c, search_h, search_w, device=self.device) * 2 - 1
        ) * self.eps

        best_fitness = float("-inf")
        num_plateaus = 0
        plateau_counter = 0
        queries_used = 0
        elite_full = None

        loop = tqdm(range(self.max_generations))

        for gen in loop:
            fitness, log_probs, population_full = self._compute_fitness(
                population_delta, x_orig, target
            )
            queries_used += self.N

            elite_idx = torch.argmax(fitness)
            elite_delta = population_delta[elite_idx : elite_idx + 1]
            elite_full = population_full[elite_idx : elite_idx + 1]
            elite_pred = log_probs[elite_idx].argmax().item()

            if elite_pred == target:
                if verbose:
                    print(f"Success at generation {gen}, queries={queries_used}")
                return elite_full, True, queries_used

            if fitness[elite_idx].item() > best_fitness + 1e-6:
                best_fitness = fitness[elite_idx].item()
                plateau_counter = 0
            else:
                plateau_counter += 1
                if plateau_counter >= self.plateau_patience:
                    num_plateaus += 1
                    plateau_counter = 0
                    if self.adaptive:
                        self._update_params(num_plateaus)

            if queries_used >= self.max_queries:
                break

            # --- build next generation (entirely in search space) ---
            probs = F.softmax(fitness / self.tau, dim=0)
            next_deltas = [elite_delta]  # elitism

            for _ in range(self.N - 1):
                idx1, idx2 = torch.multinomial(probs, 2, replacement=True)
                child = self._crossover(
                    population_delta[idx1 : idx1 + 1],
                    population_delta[idx2 : idx2 + 1],
                    fitness[idx1],
                    fitness[idx2],
                )
                child = self._mutate(child)
                next_deltas.append(child)

            population_delta = torch.cat(next_deltas, dim=0)

        if verbose:
            print(f"Failed after {queries_used} queries")
        return elite_full, False, queries_used

    # ------------------------------------------------------------------
    # Batch convenience wrapper — GenAttack is inherently per-example,
    # so this just loops attack_single and aggregates results.
    # ------------------------------------------------------------------
    def attack_batch(self, x_batch, y_batch, targets=None, verbose=False):
        adv_examples, successes, query_counts = [], [], []
        for i in range(x_batch.shape[0]):
            t = targets[i].item() if targets is not None else None
            x_adv, success, n_queries = self.attack_single(
                x_batch[i], y_batch[i].item(), target=t, verbose=verbose
            )
            adv_examples.append(x_adv)
            successes.append(success)
            query_counts.append(n_queries)
        return torch.cat(adv_examples, dim=0), successes, query_counts