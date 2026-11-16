"""Figures 2-6 and S1-S4 for the paper and supplement. Run after 01_run_analysis.py.
(Figure 1, the study area map, is made by 03_study_area_map.py.)

Common style: constrained layout; panel letters in bold at the top left of each panel; panel titles centred;
legends shared and centred below the panels; text 8.5-10 pt at the printed width (180 mm, or 140 mm for single panels).
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
import statsmodels.api as sm
sys.path.insert(0, str(Path(__file__).parent)); import socdepth as sd

ROOT = Path(__file__).resolve().parents[1]; TAB = ROOT / "outputs/tables"; FIG = ROOT / "outputs/figures"; FIG.mkdir(parents=True, exist_ok=True)
H = pd.read_csv(ROOT / "data/processed/SOC_Madagascar_harmonised_0_100cm.csv")
C = pd.read_csv(ROOT / "data/processed/calibration_with_cv_predictions.csv")
S = json.load(open(TAB / "summary_statistics.json"))

# ------------------------------------------------------------------ style
fam = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Helvetica", "TeX Gyre Heros", "Liberation Sans") if f in fam), "DejaVu Sans")
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 9.5, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "center", "axes.titlepad": 8,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "legend.handletextpad": 0.5, "legend.columnspacing": 1.6, "legend.borderaxespad": 0.2,
    "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.06, "figure.constrained_layout.w_pad": 0.06,
    "figure.constrained_layout.wspace": 0.06, "pdf.fonttype": 42, "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.06})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK, MUTED, GRID, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d55e9b", "#1a1a1a", "#5f5f5f", "#e4e4e0", "#a7aeb6"
UNIT = "Mg C ha$^{-1}$"
WRAP = {"Forests and tree plantations": "Forests and\ntree plantations", "Grasslands and degraded lands": "Grasslands and\ndegraded lands"}
wrap = lambda names: [WRAP.get(n, n) for n in names]
mm = 1 / 25.4
LETTERS = []                                     # (fig, ax, letter) for panel letters, placed after layout

def panel(ax, letter, title):
    ax.set_title(title); LETTERS.append((ax.figure, ax, letter))

def grid(ax, axis="both"):
    getattr(ax, {"x": "xaxis", "y": "yaxis"}[axis]).grid(True, color=GRID, lw=0.7) if axis != "both" else ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)

def save(fig, name):
    fig.canvas.draw()                            # run constrained layout, then place panel letters level with the titles
    r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for f, ax, letter in [t for t in LETTERS if t[0] is fig]:
        bb = inv.transform(ax.get_tightbbox(r)); tb = inv.transform(ax.title.get_window_extent(r))
        fig.text(bb[0][0], tb[0][1], letter, fontsize=13, fontweight="bold", ha="left", va="bottom")
    fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)

a_, F_ = S["final_model"]["a"], S["final_model"]["F"]; Rp = S["final_model"]["R_pool"]; RJJ = S["literature"]["R_JJ"]
ll = S["final_model"]["loglog"]
cal = H[H.role == "calibration"]; tgt = H[H.role == "target"]; mg = H[(H.role == "mangrove_paired")]; mgok = mg[mg.QC == "ok"]

# ------------------------------------------------------------------ Fig 2 depth ratio by land use
order = sd.LULC_ORDER + ["Mangroves"]
fig, ax = plt.subplots(figsize=(180 * mm, 100 * mm)); rng = np.random.default_rng(1)
for i, l in enumerate(order[::-1]):
    d = (cal if l != "Mangroves" else mgok); d = d[d.LULC == l].ratio.values; col = BLUE if l != "Mangroves" else AQUA
    ax.scatter(np.clip(d, None, 5.6), i + rng.uniform(-0.25, 0.25, len(d)), s=7, color=col, alpha=0.3, linewidths=0, zorder=2)
    ax.boxplot(np.clip(d, None, 5.6), positions=[i], vert=False, widths=0.55, showfliers=False, patch_artist=True, zorder=3,
               boxprops=dict(facecolor="none", edgecolor=INK, lw=0.9), medianprops=dict(color=INK, lw=1.8),
               whiskerprops=dict(color=INK, lw=0.8), capprops=dict(color=INK, lw=0.8))
    if l != "Mangroves":
        ax.plot(sd.lit_ratio(l, "JJveg"), i + 0.38, marker="v", ms=7, mfc=ORANGE, mec="white", mew=0.6, zorder=5)
        ax.plot(sd.lit_ratio(l, "W25"), i + 0.38, marker="D", ms=5.5, mfc=YELLOW, mec="white", mew=0.6, zorder=5)
    ax.text(5.85, i, f"{len(d)}", va="center", ha="right", fontsize=9, color=MUTED)
ax.text(5.85, len(order) - 0.35, "n", va="bottom", ha="right", fontsize=9, color=MUTED, style="italic")
ax.axvline(RJJ, color=ORANGE, lw=1.3, zorder=1); ax.axvline(sd.UNIFORM_R, color=MUTED, lw=1.1, ls="--", zorder=1); ax.axvline(1, color=MUTED, lw=0.7, ls=":")
ax.set_yticks(range(len(order))); ax.set_yticklabels(wrap(order[::-1])); ax.tick_params(axis="y", length=0)
ax.set_xlim(0.9, 5.95); ax.set_ylim(-0.6, len(order) - 0.2)
ax.set_xlabel("Depth ratio, R = SOC$_{0–100}$ / SOC$_{0–30}$   (values above 5.6 drawn at 5.6)"); grid(ax, "x")
fig.legend(handles=[Line2D([], [], color=ORANGE, lw=1.6, label=f"Global β = 0.9786 (R = {RJJ:.2f})"),
                    Line2D([], [], color=MUTED, lw=1.3, ls="--", label="Uniform density (R = 3.33)"),
                    Line2D([], [], marker="v", ls="", ms=8, mfc=ORANGE, mec="white", label="Vegetation-specific β (Jobbágy & Jackson, 2000)"),
                    Line2D([], [], marker="D", ls="", ms=6.5, mfc=YELLOW, mec="white", label="Ecosystem β (Wang et al., 2025)")],
           loc="outside lower center", ncol=2)
save(fig, "Fig2_ratio_by_lulc")

# ------------------------------------------------------------------ Fig 3 models on the calibration data
x = np.linspace(3, 185, 300)
curves = [("Uniform density", sd.UNIFORM_R * x, MUTED, "--", 1.3), ("Global β = 0.9786 (a priori)", RJJ * x, ORANGE, "-", 1.7),
          ("Fitted β / exponential, pooled", Rp * x, BLUE, "-", 1.7), ("Log–log allometric", ll["smear"] * np.exp(ll["a"] + ll["b"] * np.log(x)), MAGENTA, "-", 1.7),
          ("Exponential + background (selected)", a_ + F_ * x, INK, "-", 2.6)]
fig, axs = plt.subplots(1, 2, figsize=(180 * mm, 100 * mm))
ax = axs[0]; ax.scatter(cal.SOC30, cal.SOC100, s=9, color=GREY, alpha=0.45, linewidths=0, zorder=1)
for n_, y_, c, ls, lw in curves: ax.plot(x, y_, color=c, ls=ls, lw=lw, zorder=3)
ax.set_xlim(0, 190); ax.set_ylim(0, 480); ax.set_xlabel(f"SOC 0–30 cm ({UNIT})"); ax.set_ylabel(f"SOC 0–100 cm ({UNIT})"); grid(ax); panel(ax, "a", "Stocks")
ax = axs[1]; ax.scatter(cal.SOC30, cal.ratio, s=9, color=GREY, alpha=0.45, linewidths=0, zorder=1)
for n_, y_, c, ls, lw in curves: ax.plot(x, y_ / x, color=c, ls=ls, lw=lw, zorder=3)
g = cal.groupby(pd.qcut(cal.SOC30, 10), observed=True).agg(x=("SOC30", "median"), y=("ratio", "median"))
ax.plot(g.x, g.y, "o", ms=6.5, mfc="white", mec=INK, mew=1.3, zorder=4)
ax.set_xlim(0, 190); ax.set_ylim(0.9, 4.2); ax.set_xlabel(f"SOC 0–30 cm ({UNIT})"); ax.set_ylabel("Depth ratio, R"); grid(ax); panel(ax, "b", "Depth ratio")
fig.legend(handles=[Line2D([], [], color=c, ls=ls, lw=lw, label=n_) for n_, _, c, ls, lw in curves] +
                   [Line2D([], [], marker="o", ls="", ms=6, mfc=GREY, mec="none", alpha=0.7, label="Calibration plots"),
                    Line2D([], [], marker="o", ls="", ms=6.5, mfc="white", mec=INK, mew=1.3, label="Decile medians of observed R")],
           loc="outside lower center", ncol=3)
save(fig, "Fig3_models")

# ------------------------------------------------------------------ Fig 4 validation
Ts = {k: pd.read_csv(TAB / f).set_index("model") for k, f in [("Random 10 × 10-fold", "T3a_cv_random.csv"), ("Leave-one-source-out", "T3b_cv_leave_source_out.csv"), ("Spatial blocks", "T3c_cv_spatial_block.csv")]}
mods = [m for m in sd.MODELS if m != "UNI"][::-1]; labs = [sd.MODELS[m][0].replace(" (J&J 2000)", "").replace(" (Wang et al. 2025)", "") for m in mods]
fig, axs = plt.subplots(1, 2, figsize=(180 * mm, 112 * mm), sharey=True)
sty = {"Random 10 × 10-fold": ("o", BLUE), "Leave-one-source-out": ("s", ORANGE), "Spatial blocks": ("D", AQUA)}
for ax, col, xl, letter, title in [(axs[0], "RMSE", f"RMSE ({UNIT})", "a", "Accuracy"), (axs[1], "PBIAS", "Bias (% of observed mean)", "b", "Overall bias")]:
    ax.axhspan(mods.index("EXPb-pool") - 0.45, mods.index("EXPb-pool") + 0.45, color="#efeee8", zorder=0)
    for i, m in enumerate(mods):
        vals = [Ts[k].loc[m, col] for k in Ts]; ax.plot([min(vals), max(vals)], [i, i], color="#c4c4bf", lw=1.6, zorder=1)
        for k, (mk, c) in sty.items(): ax.plot(Ts[k].loc[m, col], i, marker=mk, ms=7, mfc=c, mec="white", mew=0.7, ls="", zorder=3)
    ax.set_xlabel(xl); grid(ax, "x"); ax.tick_params(axis="y", length=0); panel(ax, letter, title)
axs[0].set_yticks(range(len(mods))); axs[0].set_yticklabels(labs); axs[1].axvline(0, color=INK, lw=0.8)
axs[0].set_xlim(24.5, 34.5); axs[1].set_xlim(-4.5, 4.5)
u = Ts["Random 10 × 10-fold"].loc["UNI"]
fig.legend(handles=[Line2D([], [], marker=mk, ls="", ms=7, mfc=c, mec="white", label=k) for k, (mk, c) in sty.items()] +
                   [Patch(fc="#efeee8", label="Selected model")], loc="outside lower center", ncol=4)
save(fig, "Fig4_validation")

# ------------------------------------------------------------------ Fig 5 conditional bias
sel = [("B-JJ", ORANGE, "o", "Global β (a priori)"), ("R-pool", BLUE, "s", "Fitted β, pooled"), ("LL-pool", MAGENTA, "^", "Log–log"), ("EXPb-pool", INK, "D", "Exponential + background (selected)")]
fig, axs = plt.subplots(1, 2, figsize=(180 * mm, 100 * mm), gridspec_kw=dict(width_ratios=[1, 1.15]))
t = pd.read_csv(TAB / "T_S6b_loso_by_soc30_tercile.csv"); e = S["PI"]["tercile_edges"]
ax = axs[0]
for m, c, mk, l in sel:
    v = t[t.model == m].set_index("group").loc[["low", "mid", "high"], "PBIAS"].values
    ax.plot(range(3), v, "-", marker=mk, color=c, lw=2.4 if m == "EXPb-pool" else 1.6, ms=7, mec="white", mew=0.7)
ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels([f"Low\n< {e[0]:.0f}", f"Medium\n{e[0]:.0f}–{e[1]:.0f}", f"High\n> {e[1]:.0f}"])
ax.set_xlim(-0.35, 2.35); ax.set_xlabel(f"Tercile of SOC 0–30 cm ({UNIT})"); ax.set_ylabel("Bias (% of observed mean)"); grid(ax, "y"); panel(ax, "a", "By topsoil stock")
t = pd.read_csv(TAB / "T_S4_loso_by_source.csv"); srcs = t[t.model == "B-JJ"].sort_values("n").group.tolist()
nn = t[t.model == "B-JJ"].set_index("group").n
ax = axs[1]
for j, s_ in enumerate(srcs):
    vals = [t[(t.model == m) & (t.group == s_)].PBIAS.iloc[0] for m, *_ in sel]
    ax.plot([min(vals), max(vals)], [j, j], color="#c4c4bf", lw=1.6, zorder=1)
    for (m, c, mk, l), v in zip(sel, vals): ax.plot(v, j, marker=mk, ms=7, mfc=c, mec="white", mew=0.7, ls="", zorder=3)
ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(range(len(srcs)))
ax.set_yticklabels([f"{s_.replace('These_', 'Thesis ')} ({nn[s_]})" for s_ in srcs]); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Bias when the source is withheld (%)"); grid(ax, "x"); panel(ax, "b", "By withheld data source")
fig.legend(handles=[Line2D([], [], color=c, marker=mk, ms=7, mec="white", lw=2.4 if m == "EXPb-pool" else 1.6, label=l) for m, c, mk, l in sel],
           loc="outside lower center", ncol=4)
save(fig, "Fig5_conditional_bias")

# ------------------------------------------------------------------ Fig 6 harmonised product
fig, axs = plt.subplots(1, 2, figsize=(180 * mm, 100 * mm), gridspec_kw=dict(width_ratios=[1, 1.1]))
ax = axs[0]; p = C["loso_EXPb-pool"]; ax.scatter(p, C.SOC100, s=9, color=BLUE, alpha=0.35, linewidths=0, zorder=2)
ax.plot([0, 480], [0, 480], color=INK, lw=1.0, zorder=3); xs = np.linspace(14, 320, 100)
pi = S["PI"]["factors"]; edges = S["PI"]["tercile_edges"]; gi = np.digitize((xs - a_) / F_, edges); names = ["low", "mid", "high"]
ax.fill_between(xs, xs * np.array([pi[names[i]][0] for i in gi]), xs * np.array([pi[names[i]][1] for i in gi]), color=BLUE, alpha=0.12, lw=0, step="mid", zorder=1)
m_ = sd.metrics(C.SOC100, p)
ax.text(0.04, 0.96, f"RMSE = {m_['RMSE']:.1f} Mg C ha$^{{-1}}$\nBias = {m_['PBIAS']:+.1f}%\nCCC = {m_['CCC']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=9, linespacing=1.5)
ax.set_xlim(0, 330); ax.set_ylim(0, 480)
ax.set_xlabel(f"Predicted SOC 0–100 cm ({UNIT})"); ax.set_ylabel(f"Measured SOC 0–100 cm ({UNIT})"); grid(ax); panel(ax, "a", "Selected model, sources withheld")
ax = axs[1]; o = [l for l in sd.LULC_ORDER + ["Undetermined"] if (tgt.LULC == l).sum() >= 20][::-1]
for i, l in enumerate(o):
    d = tgt[tgt.LULC == l]
    for off, colname, c in [(0.19, "SOC30", GREY), (-0.19, "SOC100_pred", ORANGE)]:
        ax.boxplot(d[colname], positions=[i + off], vert=False, widths=0.32, showfliers=False, patch_artist=True, boxprops=dict(facecolor=c, edgecolor="none"),
                   medianprops=dict(color="white", lw=1.6), whiskerprops=dict(color=c, lw=1.2), capprops=dict(color=c, lw=0))
    ax.text(392, i, f"{len(d):,}", va="center", ha="right", fontsize=9, color=MUTED)
ax.text(392, len(o) - 0.45, "n", va="bottom", ha="right", fontsize=9, color=MUTED, style="italic")
ax.set_yticks(range(len(o))); ax.set_yticklabels(wrap(o)); ax.tick_params(axis="y", length=0); ax.set_xlim(0, 395); ax.set_ylim(-0.6, len(o) - 0.3)
ax.set_xlabel(f"SOC stock ({UNIT})"); grid(ax, "x"); panel(ax, "b", "Target plots")
fig.legend(handles=[Patch(fc=BLUE, alpha=0.25, label="90% prediction interval"), Line2D([], [], color=INK, lw=1.0, label="1:1 line"),
                    Patch(fc=GREY, label="Measured 0–30 cm"), Patch(fc=ORANGE, label="Predicted 0–100 cm")], loc="outside lower center", ncol=4)
save(fig, "Fig6_harmonised")

# ------------------------------------------------------------------ Fig S1 calibration support
fig, ax = plt.subplots(figsize=(150 * mm, 95 * mm))
T1 = pd.read_csv(TAB / "T1_data_summary.csv"); T1 = T1[T1.LULC.isin(sd.LULC_ORDER)].iloc[::-1].reset_index(drop=True)
y = np.arange(len(T1)); hh = 0.36
ax.barh(y + hh / 2 + 0.02, T1.n_cal, hh, color=BLUE)
ax.barh(y - hh / 2 - 0.02, T1.n_target_field, hh, color=ORANGE); ax.barh(y - hh / 2 - 0.02, T1.n_target_worldcover, hh, left=T1.n_target_field, color="#f6b89c")
for i, r in T1.iterrows():
    ax.text(r.n_cal + 18, i + hh / 2 + 0.02, f"{int(r.n_cal)}", va="center", fontsize=9); ax.text(r.n_target + 18, i - hh / 2 - 0.02, f"{int(r.n_target):,}", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(wrap(T1.LULC)); ax.set_xlabel("Number of plots"); ax.set_xlim(0, 1500); ax.tick_params(axis="y", length=0); grid(ax, "x")
fig.legend(handles=[Patch(fc=BLUE, label="Calibration plots (0–30 and 0–100 cm)"), Patch(fc=ORANGE, label="Target plots, land use from field label"),
                    Patch(fc="#f6b89c", label="Target plots, land use from WorldCover 2021")], loc="outside lower center", ncol=2)
save(fig, "FigS1_calibration_support")

# ------------------------------------------------------------------ Fig S2 residual diagnostics
X = sm.add_constant(cal.SOC30.values); o_ = sm.OLS(cal.SOC100.values, X).fit(); w_ = sm.WLS(cal.SOC100.values, X, weights=1 / cal.SOC30.values).fit()
lg = sm.OLS(np.log(cal.SOC100.values), sm.add_constant(np.log(cal.SOC30.values))).fit()
fig, axs = plt.subplots(1, 3, figsize=(180 * mm, 72 * mm))
axs[0].scatter(o_.fittedvalues, o_.resid, s=7, color=BLUE, alpha=0.35, linewidths=0); axs[0].axhline(0, color=INK, lw=0.8)
axs[0].set_xlabel(f"Fitted ({UNIT})"); axs[0].set_ylabel(f"Residual ({UNIT})"); panel(axs[0], "a", "Unweighted fit")
axs[1].scatter(w_.fittedvalues, w_.wresid, s=7, color=BLUE, alpha=0.35, linewidths=0); axs[1].axhline(0, color=INK, lw=0.8)
axs[1].set_xlabel(f"Fitted ({UNIT})"); axs[1].set_ylabel("Weighted residual"); panel(axs[1], "b", "Weighted fit (1/SOC$_{30}$)")
(osm, osr), _ = stats.probplot(lg.resid); axs[2].scatter(osm, osr, s=7, color=BLUE, alpha=0.5, linewidths=0); axs[2].plot(osm, osm * np.std(lg.resid), color=INK, lw=1.0)
axs[2].set_xlabel("Normal quantile"); axs[2].set_ylabel("Log–log residual"); panel(axs[2], "c", "Q–Q plot, log–log model")
for a in axs: grid(a)
save(fig, "FigS2_diagnostics")

# ------------------------------------------------------------------ Fig S3 mangroves
fig, axs = plt.subplots(1, 2, figsize=(180 * mm, 95 * mm))
ax = axs[0]; ax.hist(np.clip(mgok.ratio, 0, 10), bins=45, color=AQUA, alpha=0.85); ax.hist(cal.ratio, bins=np.linspace(1, 10, 46), color=BLUE, alpha=0.75)
ax.axvline(sd.UNIFORM_R, color=INK, lw=1.1, ls="--"); ax.set_xlabel("Depth ratio, R (values above 10 drawn at 10)"); ax.set_ylabel("Number of plots"); grid(ax, "y"); panel(ax, "a", "Depth ratios")
ax = axs[1]; ax.scatter(mgok.SOC30, mgok.SOC100, s=9, color=AQUA, alpha=0.4, linewidths=0); xx = np.linspace(0, 420, 50)
ax.plot(xx, sd.UNIFORM_R * xx, color=INK, ls="--", lw=1.3); ax.plot(xx, a_ + F_ * xx, color=INK, lw=2.2); ax.plot(xx, RJJ * xx, color=ORANGE, lw=1.6)
ax.set_xlim(0, 420); ax.set_ylim(0, 950); ax.set_xlabel(f"SOC 0–30 cm ({UNIT})"); ax.set_ylabel(f"SOC 0–100 cm ({UNIT})"); grid(ax); panel(ax, "b", "Mangrove plots")
fig.legend(handles=[Patch(fc=BLUE, alpha=0.75, label=f"Upland calibration plots (n = {len(cal)})"), Patch(fc=AQUA, alpha=0.85, label=f"Mangrove plots (n = {len(mgok)})"),
                    Line2D([], [], color=INK, ls="--", lw=1.3, label="Uniform density"), Line2D([], [], color=INK, lw=2.2, label="Upland model (selected)"),
                    Line2D([], [], color=ORANGE, lw=1.6, label="Global β = 0.9786")], loc="outside lower center", ncol=3)
save(fig, "FigS3_mangrove")

# ------------------------------------------------------------------ Fig S4 plot-level beta
bi = sd.beta_from_ratio(cal.ratio.values); fig, ax = plt.subplots(figsize=(140 * mm, 90 * mm))
ax.hist(np.clip(bi[np.isfinite(bi)], 0.93, 1), bins=50, color=BLUE, alpha=0.85)
ax.axvline(sd.BETA_JJ_GLOBAL, color=ORANGE, lw=1.6); ax.axvline(S["final_model"]["beta_pool"], color=INK, lw=1.6, ls="--")
ax.set_xlabel("Plot-level β (values below 0.93 drawn at 0.93)"); ax.set_ylabel("Number of plots"); grid(ax, "y")
fig.legend(handles=[Line2D([], [], color=ORANGE, lw=1.6, label="A-priori global β = 0.9786"),
                    Line2D([], [], color=INK, lw=1.6, ls="--", label=f"Fitted pooled β = {S['final_model']['beta_pool']:.4f}")], loc="outside lower center", ncol=2)
save(fig, "FigS4_plot_level_beta")
print("figures written to", FIG)
