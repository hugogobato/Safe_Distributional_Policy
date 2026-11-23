"""Generate publication-grade Phase 4 Protocol v2 figures from merged CSVs."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "phase4_v2_stress" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_dgp6_crossover():
    rows = list(csv.DictReader((ROOT / "results/phase4_v2_stress/dgp6_merged/dgp6_summary.csv").open()))
    lookup = {(r["design"], r["target"], r["method"]): r for r in rows}

    targets = [("Joint-Q-Ext", "Joint (Q + AIPW-Ext)"), ("Q-only", "Uncapped Quantile (Q-only)"), ("CVaR-cap50", "Capped CVaR (B = 50)")]
    betas = [("q80", r"$\beta = 0.80$"), ("q90", r"$\beta = 0.90$"), ("q95", r"$\beta = 0.95$")]
    ns = [1000, 2500, 5000, 10000]
    ss = [10, 25, 50, 80]

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), sharex=True, sharey=True)
    for r_idx, (t_key, t_label) in enumerate(targets):
        for c_idx, (b_key, b_label) in enumerate(betas):
            ax = axes[r_idx, c_idx]
            mat_gro = np.zeros((len(ns), len(ss)))
            mat_v1 = np.zeros((len(ns), len(ss)))
            for i, n in enumerate(ns):
                for j, s in enumerate(ss):
                    d = f"d06-n{n}-s{s:03d}-{b_key}"
                    mat_gro[i, j] = 100.0 * float(lookup[(d, t_key, "LCT-LTT-GRO")]["deployment"])
                    mat_v1[i, j] = 100.0 * float(lookup[(d, t_key, "LCT-LTT-aGRAPA")]["deployment"])
            im = ax.imshow(mat_gro, vmin=0, vmax=85, cmap="YlGnBu", aspect="auto")
            for i in range(len(ns)):
                for j in range(len(ss)):
                    g = mat_gro[i, j]
                    v1 = mat_v1[i, j]
                    txt_color = "white" if g >= 48 else "black"
                    ax.text(j, i, f"{g:.1f}%\n({v1:.1f}%)", ha="center", va="center", fontsize=8.0, color=txt_color)
            if r_idx == 0:
                ax.set_title(b_label, fontsize=11, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"{t_label}\n$n_{{\\mathrm{{cal}}}}$", fontsize=10)
            ax.set_xticks(range(len(ss)))
            ax.set_xticklabels([f"{s}%" for s in ss])
            ax.set_yticks(range(len(ns)))
            ax.set_yticklabels([f"{n:,}" for n in ns])
            if r_idx == 2:
                ax.set_xlabel("Safe-region mass $s$", fontsize=10)

    fig.suptitle(
        "DGP 6 Crossover Frontier (48 Cells × 50 Seeds = 2,400 Replications; 0.0% FWER Everywhere)\n"
        "Cell text: LCT-LTT-GRO deployment % (top) vs. v1 aGRAPA % (bottom); B1 = 0.0% across all 48 cells",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(right=0.91, top=0.90, hspace=0.14, wspace=0.08)
    cbar_ax = fig.add_axes([0.93, 0.12, 0.018, 0.76])
    fig.colorbar(im, cax=cbar_ax, label="LCT-LTT-GRO Certified Deployment (%)")
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig1_dgp6_crossover_heatmaps.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gate_g3_summary():
    rows = list(csv.DictReader((ROOT / "results/phase4_v2/v2_summary.csv").open()))
    lookup = {(r["design"], r["target"], r["method"]): r for r in rows}

    panels = [
        ("v2-d05-conflict", "Joint-Q-Ext", "DGP 5 Conflict\n(Joint Q+Ext)"),
        ("v2-d09-meps", "Joint-Q-Ext", "MEPS Proxy\n(κ=0.056, Joint)"),
        ("v2-d09-ohie", "Joint-Q-Ext", "OHIE Proxy\n(κ=0.28, Joint)"),
        ("v2-cvar-power", "CVaR-cap50", "CVaR Power\n(B=50)"),
        ("v2-cvar-trap", "CVaR-cap50", "CVaR Trap\n(B=50)"),
    ]
    methods = [
        ("LCT-LTT-GRO", "LCT-LTT-GRO (Ours)", "#1b7837"),
        ("LCT-LTT-aGRAPA", "v1 aGRAPA", "#762a83"),
        ("B2-GRO", "B2 (Mean CF)", "#2166ac"),
        ("B5+ext-GRO", "B5+ext (Fixed Cut)", "#d95f02"),
        ("B1-opra", "B1 (Separate)", "#878787"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [2.2, 1.0]})
    x = np.arange(len(panels))
    w = 0.16
    for idx, (m_key, m_label, color) in enumerate(methods):
        vals = []
        for d, t, _ in panels:
            rec = lookup.get((d, t, m_key))
            vals.append(float(rec["safe_welfare"]) if rec is not None else 0.0)
        ax1.bar(x + (idx - 2) * w, vals, width=w, label=m_label, color=color, edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([p[2] for p in panels], fontsize=9.5)
    ax1.set_ylabel("Mean Safety-Adjusted Welfare", fontsize=10)
    ax1.set_title("A. Safety-Adjusted Welfare Across Flagship Regimes (50 Seeds)", fontsize=11, fontweight="bold")
    ax1.legend(frameon=False, fontsize=8.5, ncol=2)
    ax1.grid(axis="y", alpha=0.25)

    # Panel B: FWER Failure Stress (Survival Crossing + CVaR Trap + Conflict B4)
    stress_items = [
        ("Survival Crossing\n(Joint/Q)", "v2-survival-crossing", "Joint-Q-Ext"),
        ("CVaR Severity Trap\n(CVaR B=50)", "v2-cvar-trap", "CVaR-cap50"),
        ("DGP 5 Conflict\n(Joint Q+Ext)", "v2-d05-conflict", "Joint-Q-Ext"),
    ]
    xs = np.arange(len(stress_items))
    fwer_methods = [
        ("LCT-LTT-GRO", "LCT-LTT-GRO", "#1b7837"),
        ("B5+ext-GRO", "B5+ext-GRO", "#d95f02"),
        ("B4", "B4 (Uncert.)", "#b2182b"),
    ]
    w2 = 0.24
    for idx, (m_key, m_label, color) in enumerate(fwer_methods):
        vals = [100.0 * float(lookup[(d, t, m_key)]["prefix_fwer"]) for _, d, t in stress_items]
        ax2.bar(xs + (idx - 1) * w2, vals, width=w2, label=m_label, color=color, edgecolor="black", linewidth=0.4)
    ax2.axhline(5.0, color="black", linestyle="--", linewidth=1.0, label="Nominal δ = 5%")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([s[0] for s in stress_items], fontsize=9)
    ax2.set_ylabel("Empirical Prefix FWER (%)", fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.set_title("B. Safety Violation Audit (FWER %)", fontsize=11, fontweight="bold")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig2_gate_g3_v2_summary.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_dgp6_crossover()
    plot_gate_g3_summary()
    print(f"Saved Phase 4 v2 figures to {FIG_DIR}")
