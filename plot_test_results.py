import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


ATTACK_ORDER = [
    "clean",
    "fgsm_small", "fgsm_big",
    "ifgsm_small", "ifgsm_big",
    "pgd_small",  "pgd_big",
]
 
ATTACK_LABELS = {
    "clean":       "Clean",
    "fgsm_small":  "FGSM\n(ε=2/255)",
    "fgsm_big":    "FGSM\n(ε=8/255)",
    "ifgsm_small": "I-FGSM\n(ε=2/255)",
    "ifgsm_big":   "I-FGSM\n(ε=8/255)",
    "pgd_small":   "PGD\n(ε=2/255)",
    "pgd_big":     "PGD\n(ε=8/255)",
}
 
METRICS = {
    "acc": {
        "suffix": "_acc",
        "title":  "Accuracy",
        "cmap":   "RdYlGn",   # red = bad, green = good
        "vmin":   0.0,
        "vmax":   1.0,
        "fmt":    ".2f",
    },
    "asr": {
        "suffix": "_asr",
        "title":  "Attack Success Rate (ASR)",
        "cmap":   "RdYlGn_r", # red = bad (high ASR), green = good
        "vmin":   0.0,
        "vmax":   1.0,
        "fmt":    ".2f",
    },
    "auc": {
        "suffix": "_auc",
        "title":  "AUC",
        "cmap":   "RdYlGn",
        "vmin":   0.0,
        "vmax":   1.0,
        "fmt":    ".3f",
    },
}
 
 
def _build_matrix(results: dict, suffix: str):
    """
    Build (model_names, attack_keys, matrix) from the results dict.
    Skips attack keys that have no data for *any* model.
    """
    model_names = list(results.keys())
 
    # only keep attacks that exist for at least one model
    valid_attacks = []
    for atk in ATTACK_ORDER:
        key = f"test_{atk}{suffix}"
        if any(key in results[m] for m in model_names):
            valid_attacks.append(atk)
 
    matrix = np.full((len(model_names), len(valid_attacks)), np.nan)
    for i, model in enumerate(model_names):
        for j, atk in enumerate(valid_attacks):
            key = f"test_{atk}{suffix}"
            if key in results[model]:
                matrix[i, j] = results[model][key]
 
    return model_names, valid_attacks, matrix
 
 
def plot_heatmaps(
    json_path: str | Path,
    output_dir: str | Path = ".",
    figsize_per_cell: tuple[float, float] = (1.4, 0.7),
    dpi: int = 150,
    save: bool = True,
    show: bool = True,
) -> dict[str, plt.Figure]:
    """
    Load a JSON file of model results and produce one heatmap per metric
    (accuracy, ASR, AUC).
 
    JSON structure expected:
        {
            "model_name_1": { "test_clean_acc": ..., "test_fgsm_big_acc": ..., ... },
            "model_name_2": { ... },
            ...
        }
 
    Parameters
    ----------
    json_path       : path to the consolidated JSON file
    output_dir      : directory where PNGs are saved (created if missing)
    figsize_per_cell: (width, height) in inches per cell; total fig size is computed automatically
    dpi             : resolution
    save            : write PNGs to output_dir
    show            : call plt.show() after generating
 
    Returns
    -------
    dict mapping metric key ("acc", "asr", "auc") -> matplotlib Figure
    """
    json_path  = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    with open(json_path) as f:
        results = json.load(f)
 
    figures = {}
 
    for metric_key, cfg in METRICS.items():
        model_names, valid_attacks, matrix = _build_matrix(results, cfg["suffix"])
 
        n_models  = len(model_names)
        n_attacks = len(valid_attacks)
 
        w = max(6, n_attacks * figsize_per_cell[0] + 2.5)
        h = max(3, n_models  * figsize_per_cell[1] + 1.5)
 
        fig, ax = plt.subplots(figsize=(w, h))
 
        im = ax.imshow(
            matrix,
            cmap=cfg["cmap"],
            vmin=cfg["vmin"],
            vmax=cfg["vmax"],
            aspect="auto",
        )
 
        # ── axes ──────────────────────────────────────────────────────────────
        ax.set_xticks(range(n_attacks))
        ax.set_xticklabels(
            [ATTACK_LABELS.get(a, a) for a in valid_attacks],
            fontsize=9, ha="center",
        )
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(model_names, fontsize=9)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
 
        # ── cell annotations ──────────────────────────────────────────────────
        fmt = cfg["fmt"]
        thresh_lo = cfg["vmin"] + (cfg["vmax"] - cfg["vmin"]) * 0.25
        thresh_hi = cfg["vmin"] + (cfg["vmax"] - cfg["vmin"]) * 0.75
 
        for i in range(n_models):
            for j in range(n_attacks):
                val = matrix[i, j]
                if np.isnan(val):
                    txt, color = "–", "#888"
                else:
                    txt = f"{val:{fmt}}"
                    # pick contrasting text color
                    norm_val = (val - cfg["vmin"]) / (cfg["vmax"] - cfg["vmin"])
                    cmap_obj  = plt.get_cmap(cfg["cmap"])
                    bg_rgba   = cmap_obj(norm_val)
                    luminance = 0.299*bg_rgba[0] + 0.587*bg_rgba[1] + 0.114*bg_rgba[2]
                    color     = "black" if luminance > 0.45 else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color=color, fontweight="bold")
 
        # ── colorbar ──────────────────────────────────────────────────────────
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
 
        # ── title & layout ────────────────────────────────────────────────────
        ax.set_title(cfg["title"], fontsize=13, fontweight="bold", pad=18)
        fig.tight_layout()
 
        if save:
            out_path = output_dir / f"heatmap_{metric_key}_fgsm_training.png"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {out_path}")
 
        figures[metric_key] = fig
 
    if show:
        plt.show()
 
    return figures


if __name__ == "__main__":

    figs = plot_heatmaps(
        json_path="fgsm_test_results.json",  # your consolidated file
        output_dir="comparison_images",          # where PNGs are saved
        show=False,                 # call plt.show()
    )