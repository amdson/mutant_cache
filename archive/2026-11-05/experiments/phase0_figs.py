"""
phase0_figs.py -- figures for PHASE0_RESULTS.md from the two phase-0 CSVs.

    cd ~/mutant_cache
    ~/anaconda3/envs/mutant_cache/bin/python experiments/phase0_figs.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGS = "/home/andrew/mutant_cache/experiments/figs"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 5, "savefig.dpi": 160, "savefig.bbox": "tight",
})

deg = pd.read_csv("/home/andrew/mutant_cache/experiments/phase0_degradation_summary.csv")
grid = pd.read_csv("/home/andrew/mutant_cache/experiments/phase0_runtime_grid.csv")
nsw = pd.read_csv("/home/andrew/mutant_cache/experiments/phase0_runtime_nsweep.csv")
ORDER = ["GB1", "ubiquitin", "avGFP", "PTEN"]


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points",
                color=color, fontsize=8, va="center")


def spread(vals, gap=0.025):
    """Vertical offsets (points) so end labels at nearly equal y don't collide."""
    order = np.argsort(vals)
    span = max(np.ptp(vals), 1e-9)
    off = np.zeros(len(vals))
    last = -np.inf
    for i in order:
        want = vals[i]
        if want - last < gap * span:
            off[i] = (last + gap * span - want) / span * 60
            last = last + gap * span
        else:
            last = want
    return off


# --- fig 1: cosine vs mutation count ---------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
for (col, title, ylab) in [("cos_mut", "mutated positions", "cosine vs exact"),
                           ("cos_unmut", "unmutated positions", ""),
                           ("cos_pool", "mean-pooled sequence", "")]:
    ax = axes[["cos_mut", "cos_unmut", "cos_pool"].index(col)]
    ends = np.array([deg[deg.protein == p].sort_values("n_mut")[col].iloc[-1]
                     for p in ORDER])
    offs = spread(ends, gap=0.06)
    for i, p in enumerate(ORDER):
        g = deg[deg.protein == p].sort_values("n_mut")
        ax.plot(g.n_mut, g[col], "-o", color=SER[i], label=f"{p} ({g.L.iloc[0]} aa)")
        label_end(ax, g.n_mut.iloc[-1], g[col].iloc[-1], p, SER[i], dy=offs[i])
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("substitutions")
    ax.set_title(title, color=INK2, fontsize=9)
    ax.set_xlim(0.9, 90)
    if ylab:
        ax.set_ylabel(ylab)
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Cache vs exact mutant forward: final-hidden cosine (ESM-C 300M, fp32)",
             fontsize=10)
fig.savefig(f"{FIGS}/fig1_cosine_vs_nmut.png")

# --- fig 2: error vs density -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
for i, p in enumerate(ORDER):
    g = deg[deg.protein == p].sort_values("density")
    axes[0].plot(g.density, 1 - g.cos_mut, "-o", color=SER[i], label=p)
    axes[1].plot(g.density, 1 - g.cos_unmut, "-o", color=SER[i], label=p)
for ax, t in zip(axes, ["1 - cosine, mutated positions",
                        "1 - cosine, unmutated positions"]):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mutation density  (n_mut / L)")
    ax.set_title(t, color=INK2, fontsize=9)
axes[0].set_ylabel("approximation error")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Unmutated-position error tracks mutation density across proteins; "
             "mutated-position error has a per-protein floor", fontsize=10)
fig.savefig(f"{FIGS}/fig2_error_vs_density.png")

# --- fig 3: LLR rank agreement ---------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
for i, p in enumerate(ORDER):
    g = deg[deg.protein == p].sort_values("n_mut")
    axes[0].plot(g.n_mut, g.spearman, "-o", color=SER[i], label=p)
    axes[1].plot(g.n_mut, g.llr_gap_abs, "-o", color=SER[i], label=p)
axes[0].axhline(1.0, color=GRID, lw=1)
axes[0].set_ylabel("Spearman(cache, exact)")
axes[0].set_title("masked-marginal LLR rank agreement", color=INK2, fontsize=9)
axes[1].set_ylabel("mean |gap| (nats)")
axes[1].set_yscale("log")
axes[1].set_title("masked-marginal LLR absolute gap", color=INK2, fontsize=9)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("substitutions")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Zero-shot score agreement, 64 random mutants per cell", fontsize=10)
fig.savefig(f"{FIGS}/fig3_llr_agreement.png")

# --- fig 4: runtime --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
g = grid[grid.dtype == "bf16"]
for i, L in enumerate(sorted(g.L.unique())):
    s = g[g.L == L].sort_values("n_mut")
    axes[0].plot(s.n_mut, s.speedup_amort, "-o", color=SER[i], label=f"L={L}")
    label_end(axes[0], s.n_mut.iloc[-1], s.speedup_amort.iloc[-1], f"{L}", SER[i])
axes[0].axhline(1.0, color="#e34948", lw=1.2, ls="--")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel("substitutions per mutant")
axes[0].set_ylabel("speedup vs full forward")
axes[0].set_title(f"amortized over N={int(g.N.iloc[0])} mutants (bf16)", color=INK2, fontsize=9)
axes[0].legend(frameon=False, fontsize=8)

for i, (L, K) in enumerate([(512, 1), (512, 5), (512, 20)]):
    s = nsw[(nsw.L == L) & (nsw.n_mut == K)].sort_values("N")
    axes[1].plot(s.N, s.speedup_amort, "-o", color=SER[i], label=f"K={K}")
axes[1].axhline(1.0, color="#e34948", lw=1.2, ls="--")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("mutants per reference (N)")
axes[1].set_ylabel("speedup vs full forward")
axes[1].set_title("amortization of the reference pass, L=512", color=INK2, fontsize=9)
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle("Variant cache throughput, ESM-C 300M on one A6000", fontsize=10)
fig.savefig(f"{FIGS}/fig4_runtime.png")
print("wrote figures to", FIGS)
