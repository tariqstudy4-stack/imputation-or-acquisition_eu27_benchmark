"""Regenerates figures/fig_experiment_design.png.

Standalone script (no dependency on the rest of code/): draws the
experimental design / reproducibility pipeline diagram described in
README.md Fig. A1, from a fixed content spec below, so anyone can
reproduce or restyle the figure without re-deriving the pipeline from
the result files.

Style: "amber blueprint" -- warm accent color, rounded boxes with a
subtle drop shadow, numbered stage badges in the left margin. This is
a deliberately different visual style from the companion Scope 3
repository's fig_experiment_design.png (slate/teal, sharp corners,
left accent bars); the two figures describe different pipelines and
are not meant to look interchangeable.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["font.family"] = "DejaVu Serif"

INK = "#2b2b2b"
BOX_FILL = "#f7f5f2"
BOX_EDGE = "#4a4a4a"
AMBER_FILL = "#fdf1dc"
AMBER_EDGE = "#a8641f"
AMBER_BADGE = "#b6752d"
SLATE_BADGE = "#5a5a5a"
SHADOW = "#d9d5cd"

fig, ax = plt.subplots(figsize=(9.0, 10.2), dpi=220)
ax.set_xlim(-0.75, 10)
ax.set_ylim(0, 11.4)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(cx, cy, w, h, text, highlight=False, fontsize=9.3):
    fill = AMBER_FILL if highlight else BOX_FILL
    edge = AMBER_EDGE if highlight else BOX_EDGE
    lw = 2.0 if highlight else 1.3
    shadow = FancyBboxPatch(
        (cx - w / 2 + 0.06, cy - h / 2 - 0.06), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=0, facecolor=SHADOW, zorder=1,
    )
    ax.add_patch(shadow)
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             color=INK, zorder=3, linespacing=1.45)


def badge(y, n, highlight=False):
    x = -0.35
    c = Circle((x, y), 0.19, facecolor=(AMBER_BADGE if highlight else SLATE_BADGE),
                edgecolor="none", zorder=4, clip_on=False)
    ax.add_patch(c)
    ax.text(x, y, str(n), ha="center", va="center", fontsize=8.5,
             color="white", weight="bold", zorder=5, clip_on=False)


def arrow(p_from, p_to, rad=0.0):
    a = FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=13,
        color=INK, linewidth=1.3, shrinkA=2, shrinkB=2,
        connectionstyle=f"arc3,rad={rad}", zorder=2,
    )
    ax.add_patch(a)


def fan(src_y, dst_y, xs_dst, x_src=5.0):
    """Straight fan-out/fan-in arrows between a center point and several
    x positions on the row below (or above); avoids the tangent-flip
    artifacts that large-radius curves produce when points converge."""
    for x in xs_dst:
        arrow((x_src, src_y), (x, dst_y), rad=0.0)


# Title
ax.text(4.6, 11.05, "Experimental design and reproducibility pipeline",
         ha="center", va="center", fontsize=16.5, weight="bold", color=INK)
ax.text(4.6, 10.6, "Gap-filling benchmark for live, partial life cycle "
                    "inventories over the EU-27 technosphere",
         ha="center", va="center", fontsize=10.5, style="italic", color="#555")

# Row 1: data sources
box(2.3, 9.6, 4.3, 1.15,
    "Eurostat naio_10_cp1750\nEU-27 symmetric input-output table\n"
    "64-industry NACE Rev.2, 2022")
box(6.9, 9.6, 4.3, 1.15,
    "Eurostat env_ac_ainah_r2\nAir emissions accounts\nby NACE Rev.2, 2022")
badge(9.6, 1)
arrow((2.3, 9.6 - 0.575), (4.6, 8.1 + 0.5), rad=-0.15)
arrow((6.9, 9.6 - 0.575), (4.6, 8.1 + 0.5), rad=0.15)

# Row 2: data_build
box(4.6, 8.1, 9.3, 1.0,
    "data_build/  —  A_eu.npy (technosphere), B_eu.npy (extension), "
    "C_eu.npy (characterization)\n19 manufacturing targets, "
    "171 pairwise comparisons / indicator", fontsize=9.0)
badge(8.1, 2)
arrow((4.6, 8.1 - 0.5), (4.6, 6.55 + 0.675))

# Row 3: missingness design (highlight)
box(4.6, 6.55, 9.5, 1.55,
    "Missingness design (imposed on A, B)\n"
    "MCAR (random loss)  ·  FLOW (sensor failure, row-structured)\n"
    "BLOCK (supplier non-disclosure, column-structured)\n"
    "rates: 5, 10, 20, 40%   ×   3 seeds / condition",
    highlight=True, fontsize=8.6)
badge(6.55, 3, highlight=True)

# Row 4: five methods
methods = [
    ("PEDIGREE", "sibling proxy +\nlognormal MC\n(GSD 2.0 / 1.8)"),
    ("KNN", "donor imputation\n(k=10) +\nresample MC"),
    ("SOFTIMPUTE", "low-rank matrix\ncompletion\n(soft-SVD)"),
    ("VAE", "column autoencoder,\ntrained per condition\n+ decoder MC"),
    ("GNN", "message passing on\ntechnosphere graph\n+ MC dropout"),
]
xs = [0.5, 2.55, 4.6, 6.65, 8.7]
for x, (name, desc) in zip(xs, methods):
    box(x, 5.0, 1.9, 1.6, f"{name}\n{desc}", fontsize=7.7)
badge(5.0, 4)
fan(6.55 - 0.775, 5.0 + 0.8, xs, x_src=4.6)

# Row 5: Leontief propagation (highlight)
box(4.6, 3.35, 9.5, 1.4,
    "Leontief propagation + characterization on every repaired inventory\n"
    "Metrics: impact-level MdAPE  ·  95% Monte Carlo interval coverage\n"
    "material ranking flips (gap > 5%)", highlight=True, fontsize=8.6)
badge(3.35, 5, highlight=True)
fan(5.0 - 0.8, 3.35 + 0.7, xs, x_src=4.6)

# Row 6: outputs
box(2.3, 2.0, 4.3, 1.0,
    "results/results_final_eu*.csv\n(main 3×4 pattern-rate grid)", fontsize=9.0)
box(6.9, 2.0, 4.3, 1.0,
    "code/gsd_sensitivity.py\n→ results/gsd_sensitivity_eu.csv\n"
    "(GSD swept 1.2 – 3.0)", fontsize=9.0)
badge(2.0, 6)
arrow((4.6, 3.35 - 0.7), (2.3, 2.0 + 0.5), rad=0.15)
arrow((4.6, 3.35 - 0.7), (6.9, 2.0 + 0.5), rad=-0.15)

# Row 7: figures
box(4.6, 0.65, 9.3, 1.0,
    "code/make_figures.py\n"
    "→ figures/chart_error_*.png, chart_coverage_*.png, chart_flips_pedigree.png,\n"
    "fig_gsd_sensitivity.png, fig_eu_full_benchmark.png", fontsize=9.0)
badge(0.65, 7)
arrow((2.3, 2.0 - 0.5), (4.6, 0.65 + 0.5), rad=-0.15)
arrow((6.9, 2.0 - 0.5), (4.6, 0.65 + 0.5), rad=0.15)

plt.tight_layout()
plt.savefig("fig_experiment_design.png", dpi=220, facecolor="white")
print("saved fig_experiment_design.png")
