import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ── configuración ────────────────────────────────────────────────────────────
CSV_PATH  = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/dL_vs_thr_exp_01.csv'
   # cambiá esto por la ruta a tu archivo
CMAP_LINES = "viridis"   # colormap continuo para curvas por thr1
NROWS, NCOLS = 7, 3      # grilla para subplots de thr2

# ── carga ────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
required = {"thr1", "thr2", "thr3", "dL1", "dL2", "dL3", "reduction_pars_coll"}
assert required.issubset(df.columns), f"Faltan columnas: {required - set(df.columns)}"

# df = df.rename(columns={"t1": "thr1", "t2": "thr2", "t3": "thr3"})
df["dl_sum"] = df["dL1"] + df["dL2"] + df["dL3"]

thr1_vals = sorted(df["thr1"].unique())
thr2_vals = sorted(df["thr2"].unique())

# colormap continuo normalizado sobre thr1
norm      = mcolors.Normalize(vmin=thr1_vals[0], vmax=thr1_vals[-1])
cmap_obj  = cm.get_cmap(CMAP_LINES)
color_of  = lambda v: cmap_obj(norm(v))

# mappable para colorbar
sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
sm.set_array([])


# ── figura 1: dl1 = f(thr1) ──────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(6, 4))
d1 = df.drop_duplicates("thr1").sort_values("thr1")
ax.plot(d1["thr1"], d1["dL1"], lw=1.5, color="#3266ad", marker="o", ms=4)
ax.set_xlabel("thr1", fontsize=11)
ax.set_ylabel("dl1", fontsize=11)
ax.set_title("dl1  =  f(thr1)", fontsize=12)
ax.grid(True, lw=0.4, alpha=0.5)
fig1.tight_layout()
fig1.savefig("plot_dl1.svg", dpi=150, bbox_inches="tight", format='svg')


# ── figura 2: dl2 = f(thr2), curvas por thr1 ─────────────────────────────────
fig2, ax = plt.subplots(figsize=(7, 4))
for thr1v in thr1_vals:
    sub = df[df["thr1"] == thr1v].groupby("thr2", as_index=False)["dL2"].mean()
    ax.plot(sub["thr2"], sub["dL2"], lw=1.2, color=color_of(thr1v), marker="o", ms=2)
ax.set_xlabel("thr2", fontsize=11)
ax.set_ylabel("dl2", fontsize=11)
ax.set_title("dl2  =  f(thr2)  —  curvas por thr1", fontsize=12)
ax.grid(True, lw=0.4, alpha=0.5)
cb = fig2.colorbar(sm, ax=ax, pad=0.02)
cb.set_label("thr1", fontsize=9)
fig2.tight_layout()
fig2.savefig("plot_dl2.svg", dpi=150, bbox_inches="tight", format='svg')


# ── helper: figura 7×3 con subplots por thr2, curvas por thr1 ────────────────
def plot_by_thr2(z_col, filename, fig_title):
    fig, axes = plt.subplots(NROWS, NCOLS,
                             figsize=(4.5 * NCOLS, 3 * NROWS),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for idx, thr2v in enumerate(thr2_vals):
        ax = axes_flat[idx]
        sub2 = df[df["thr2"] == thr2v]
        for thr1v in thr1_vals:
            sub = sub2[sub2["thr1"] == thr1v].groupby("thr3", as_index=False)[z_col].mean()
            if sub.empty:
                continue
            ax.plot(sub["thr3"], sub[z_col], lw=1.0,
                    color=color_of(thr1v), marker="o", ms=2)
        ax.set_title(f"thr2={thr2v:.3g}", fontsize=8, pad=3)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=7)
        col_idx = idx % NCOLS
        row_idx = idx // NCOLS
        if col_idx == 0:
            ax.set_ylabel(z_col, fontsize=8)
        if row_idx == NROWS - 1 or idx >= len(thr2_vals) - NCOLS:
            ax.set_xlabel("thr3", fontsize=8)

    # ocultar subplots vacíos
    for idx in range(len(thr2_vals), NROWS * NCOLS):
        axes_flat[idx].set_visible(False)

    # colorbar global
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02, shrink=0.6)
    cb.set_label("thr1", fontsize=10)

    fig.suptitle(fig_title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])
    fig.savefig(filename, dpi=150, bbox_inches="tight", format='svg')


# ── figura 3: dl3 ─────────────────────────────────────────────────────────────
plot_by_thr2("dL3",  "plot_dl3.svg",
             "dl3  =  f(thr3)  —  subplots por thr2,  curvas por thr1")

# ── figura 4: prop ────────────────────────────────────────────────────────────
plot_by_thr2("reduction_pars_coll", "plot_prop.svg",
             "prop  =  f(thr3)  —  subplots por thr2,  curvas por thr1")

# ── figura 5: dl1+dl2+dl3 ─────────────────────────────────────────────────────
plot_by_thr2("dl_sum", "plot_dl_sum.svg",
             "dl1+dl2+dl3  =  f(thr3)  —  subplots por thr2,  curvas por thr1")

plt.show()
# print("Guardado: plot_dl1.png  plot_dl2.png  plot_dl3.png  plot_prop.png")
