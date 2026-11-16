"""Prepare data, evaluate depth-harmonisation models, run statistical tests, predict SOC 0-100 cm.

Usage: python src/01_run_analysis.py data/raw/SOCv5.xlsx
Outputs: outputs/tables/*.csv, data/processed/*.csv|xlsx
"""
import sys, json, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import pyproj
sys.path.insert(0, str(Path(__file__).parent))
import socdepth as sd
warnings.filterwarnings("ignore")

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/SOCv5.xlsx")
ROOT = Path(__file__).resolve().parents[1]; TAB = ROOT / "outputs/tables"; PROC = ROOT / "data/processed"
TAB.mkdir(parents=True, exist_ok=True); PROC.mkdir(parents=True, exist_ok=True)
SEED, NREP, NFOLD, NBOOT = 2026, 10, 10, 2000
FINAL = "EXPb-pool"       # model used to build the harmonised dataset (justified by the evaluation; see paper)
rng = np.random.default_rng(SEED)

# =============================================================== 1. data preparation
df = pd.read_excel(RAW)
df = df.rename(columns={"Type": "Database", "Base": "Source", "Cluster": "Year", "Type.1": "ValFlag"})
df["Source"] = df.Source.astype(str)
df["LULC_detail"] = df.LU.map(sd.LULC_MAP)
assert df.LULC_detail.notna().all()
df["LULC_field"] = df.LULC_detail.map(sd.LULC_GROUP)
# land cover of plots without a field label: ESA WorldCover 2021 (Zanaga et al., 2022), sampled at the plot location
wc = pd.read_csv(RAW.parent / "soc_plots_worldcover2021.csv")[["num", "WC2021", "WC2021_mode50m"]]
df = df.merge(wc, on="num", how="left", validate="one_to_one"); assert df.WC2021.notna().all()
df["LULC_WC"] = df[sd.WC_COL].map(sd.WC_GROUP).fillna("Undetermined")
und = df.LULC_field == "Undetermined"
df["LULC"] = np.where(und, df.LULC_WC, df.LULC_field)
df.loc[und & (df.LULC == "Mangroves"), "LULC"] = "Undetermined"          # an upland plot cannot be re-labelled as mangrove
df["LULC_source"] = np.where(und, np.where(df.LULC == "Undetermined", "none", "WorldCover 2021"), "field")
# agreement of WorldCover with the field labels (upland plots with a field label), for both sampling options
lab = df[~und & (df.LULC_field != "Mangroves")]
agree = {c: float((lab[c].map(sd.WC_GROUP).fillna("Undetermined") == lab.LULC_field).mean()) for c in ["WC2021", "WC2021_mode50m"]}
conf = pd.crosstab(lab.LULC_field, lab.LULC_WC).reindex(index=sd.LULC_ORDER, columns=sd.LULC_ORDER + ["Undetermined"]).fillna(0).astype(int)
conf["n"] = conf.sum(1); conf["agreement_pct"] = [100 * conf.loc[l, l] / conf.loc[l, "n"] for l in conf.index]; conf.to_csv(TAB / "T_S15_worldcover_vs_field.csv")
wct = df[und].assign(kind=np.where(df[und].SOC100.notna(), "paired_excluded", "target")).groupby([sd.WC_COL, "kind"]).size().unstack(fill_value=0); wct["class"] = [sd.WC_NAMES[k] for k in wct.index]; wct["group"] = [sd.WC_GROUP.get(k, "Undetermined") for k in wct.index]
wct.to_csv(TAB / "T_S14_worldcover_assignment.csv"); print(conf.to_string()); print(wct.to_string()); print("agreement", agree)
df["KG"] = [sd.koppen(a, b) for a, b in zip(df.Latitude, df.Longitude)]       # Koppen-Geiger class 1986-2010
assert df.KG.notna().all()
df["has100"] = df.SOC100.notna()
df["ratio"] = df.SOC100 / df.SC30
qc = pd.Series("not_applicable", index=df.index)
h = df.has100
qc[h] = "ok"
qc[h & (df.ratio < 1 - 1e-6)] = "impossible_SOC100_lt_SOC30"
qc[h & np.isclose(df.ratio, 1, atol=1e-6)] = "SOC100_equals_SOC30"
qc[h & (df.Source == "PERR-FH")] = "anomalous_source"
qc[h & (df.LULC == "Mangroves") & (df.ratio > 10)] = "implausible_ratio_gt10"
df["QC"] = qc
df["role"] = np.select([~h, h & (df.LULC == "Mangroves"), h & (df.QC == "ok")],
                       ["target", "mangrove_paired", "calibration"], "paired_excluded")
cal = df[df.role == "calibration"].copy().reset_index(drop=True)
tgt = df[df.role == "target"].copy()
tr = pyproj.Transformer.from_crs(4326, 32738, always_xy=True)
cal["E"], cal["N"] = tr.transform(cal.Longitude.values, cal.Latitude.values)

flow = df.groupby(["role", "QC"]).size().rename("n").reset_index(); flow.to_csv(TAB / "T_S1_qc_flow.csv", index=False)
print(flow.to_string())
key = df.groupby(["LULC_field", "LULC_detail", "LU"]).agg(n_total=("SC30", "size"), n_paired=("has100", "sum")).reset_index()
key.to_csv(TAB / "T_S2_lulc_key.csv", index=False)
kgt = pd.crosstab(df.KG, df.role).reindex(sd.KG_ORDER).fillna(0).astype(int); kgt["all_plots"] = kgt.sum(1)
kgt.to_csv(TAB / "T_S13_koppen_by_role.csv")

# Table 1: data summary by LULC
def q(s, p): return np.nanpercentile(s, p) if len(s) else np.nan
rows = []
for l in sd.LULC_ORDER + ["Undetermined", "Mangroves"]:
    c = cal[cal.LULC == l]; t = tgt[tgt.LULC == l]; nf = int((t.LULC_source == "field").sum()); m = df[(df.LULC == l) & (df.role == "mangrove_paired")]
    c = c if l != "Mangroves" else m[m.QC == "ok"]
    rows.append(dict(LULC=l, n_cal=len(c), n_target=len(t), n_target_field=nf, n_target_worldcover=len(t) - nf,
                     SOC30_cal_med=q(c.SC30, 50), SOC30_cal_p5=q(c.SC30, 5), SOC30_cal_p95=q(c.SC30, 95),
                     SOC100_cal_med=q(c.SOC100, 50), ratio_med=q(c.ratio, 50), ratio_q1=q(c.ratio, 25), ratio_q3=q(c.ratio, 75),
                     ratio_gt_uniform_pct=100 * (c.ratio > sd.UNIFORM_R).mean() if len(c) else np.nan,
                     SOC30_tgt_med=q(t.SC30, 50), SOC30_tgt_p5=q(t.SC30, 5), SOC30_tgt_p95=q(t.SC30, 95)))
T1 = pd.DataFrame(rows); T1.to_csv(TAB / "T1_data_summary.csv", index=False)
det = cal.groupby("LULC_detail").agg(group=("LULC", "first"), n_cal=("SC30", "size"), n_sources=("Source", "nunique"), ratio_median=("ratio", "median"),
                                     ratio_q1=("ratio", lambda s: s.quantile(.25)), ratio_q3=("ratio", lambda s: s.quantile(.75))).reindex(sd.LULC_DETAIL_ORDER)
det["R_fit"] = [sd.wls_ratio(cal.SC30[cal.LULC_detail == l].values, cal.SOC100[cal.LULC_detail == l].values) for l in det.index]
det["n_target_field"] = [int(((df.role == "target") & (df.LULC_detail == l)).sum()) for l in det.index]; det.to_csv(TAB / "T_S16_detailed_classes.csv")
print(T1.round(2).to_string())

# =============================================================== 2. inference on depth distribution
out = {"worldcover_agreement": agree, "worldcover_column": sd.WC_COL}
# 2a ratio differences among LULC
groups = [cal.ratio[cal.LULC == l].values for l in sd.LULC_ORDER if (cal.LULC == l).sum() > 0]
H, p = stats.kruskal(*groups); k = len(groups); n = len(cal)
out["kruskal"] = dict(H=H, df=k - 1, p=p, eps2=(H - k + 1) / (n - k))
# pairwise Mann-Whitney (Holm)
pw = []
for a, b in itertools.combinations(sd.LULC_ORDER, 2):
    x, y = cal.ratio[cal.LULC == a], cal.ratio[cal.LULC == b]
    U, pp = stats.mannwhitneyu(x, y); pw.append(dict(A=a, B=b, nA=len(x), nB=len(y), medA=x.median(), medB=y.median(), U=U, p=pp))
pw = pd.DataFrame(pw); pw["p_holm"] = multipletests(pw.p, method="holm")[1]; pw.to_csv(TAB / "T_S3_pairwise_ratio_lulc.csv", index=False)
# 2b plot-level beta and tests against literature hypotheses
cal["beta_i"] = sd.beta_from_ratio(cal.ratio.values)
def boot_ci(x, y, f, B=NBOOT):
    idx = rng.integers(0, len(x), (B, len(x))); v = np.array([f(x[i], y[i]) for i in idx]); return np.percentile(v, [2.5, 97.5])
rows = []
for l in ["All"] + sd.LULC_ORDER:
    c = cal if l == "All" else cal[cal.LULC == l]
    x, y = c.SC30.values, c.SOC100.values
    R = sd.wls_ratio(x, y); ci = boot_ci(x, y, sd.wls_ratio)
    b = sd.beta_from_ratio([R])[0]; bci = sd.beta_from_ratio(ci)
    for scheme in ["JJ", "JJveg", "W25"]:
        pass
    r0 = {s: sd.lit_ratio(l if l != "All" else "Undetermined", s) for s in ["JJ", "JJveg", "W25"]}
    w = {s: stats.wilcoxon(c.ratio.values - r0[s]).pvalue for s in r0}
    rows.append(dict(LULC=l, n=len(c), R_fit=R, R_ols=sd.ls_ratio(x, y), R_lo=ci[0], R_hi=ci[1], beta_fit=b, beta_lo=bci[0], beta_hi=bci[1],
                     k_per_cm=-np.log(b), ratio_median=c.ratio.median(), ratio_mean=c.ratio.mean(),
                     beta_i_median=np.nanmedian(c.beta_i), pct_ratio_gt_uniform=100 * (c.ratio > sd.UNIFORM_R).mean(),
                     R_JJ=r0["JJ"], p_vs_JJ=w["JJ"], R_JJveg=r0["JJveg"], p_vs_JJveg=w["JJveg"], R_W25=r0["W25"], p_vs_W25=w["W25"]))
T2 = pd.DataFrame(rows)
for c_ in ["p_vs_JJ", "p_vs_JJveg", "p_vs_W25"]: T2[c_ + "_holm"] = multipletests(T2[c_], method="holm")[1]
T2.to_csv(TAB / "T2_fitted_vs_literature_beta.csv", index=False); print(T2.round(4).to_string())
# 2c nested-model tests (ANCOVA) and diagnostics
c2 = cal.assign(LULC=pd.Categorical(cal.LULC, sd.LULC_ORDER), lx=np.log(cal.SC30), ly=np.log(cal.SOC100))
# variance function: |OLS residual| ~ SOC30^g  (g = 0.5 means Var ~ SOC30 and justifies weights 1/SOC30)
o1 = smf.ols("SOC100 ~ SC30", c2).fit()
vf = sm.OLS(np.log(np.abs(o1.resid) + 1e-9), sm.add_constant(np.log(c2.SC30))).fit()
out["variance_function"] = dict(g=vf.params.iloc[1], g_lo=vf.conf_int().iloc[1, 0], g_hi=vf.conf_int().iloc[1, 1], BP_p_ols=het_breuschpagan(o1.resid, o1.model.exog)[1])
w = 1 / c2.SC30
f0 = smf.wls("SOC100 ~ 0 + SC30", c2, weights=w).fit(); f1 = smf.wls("SOC100 ~ SC30", c2, weights=w).fit()
f2 = smf.wls("SOC100 ~ SC30 + C(LULC)", c2, weights=w).fit(); f3 = smf.wls("SOC100 ~ SC30 * C(LULC)", c2, weights=w).fit()
g1 = smf.ols("ly ~ lx", c2).fit(); g2 = smf.ols("ly ~ lx + C(LULC)", c2).fit(); g3 = smf.ols("ly ~ lx * C(LULC)", c2).fit()
def ftest(a, b):
    F, p, d = b.compare_f_test(a); return dict(F=F, p=p, df_num=d, df_den=b.df_resid)
r1 = f1.get_robustcov_results("HC3")
out["nested"] = {"lin_intercept_vs_origin": ftest(f0, f1), "lin_LULC_intercepts": ftest(f1, f2), "lin_LULC_slopes": ftest(f2, f3),
                 "log_LULC_intercepts": ftest(g1, g2), "log_LULC_slopes": ftest(g2, g3)}
out["linear"] = dict(a=f1.params["Intercept"], F=f1.params["SC30"], a_se_HC3=r1.bse[0], F_se_HC3=r1.bse[1], a_p_HC3=r1.pvalues[0],
                     R2=f1.rsquared, BP_p_wls=het_breuschpagan(f1.wresid, f1.model.exog)[1])
Fs = f1.params["SC30"]; a = f1.params["Intercept"]
kk, cb = sd.expb_to_depth_params(a, Fs)
def _bp(x, y):
    f_ = sm.WLS(y, sm.add_constant(x), weights=1 / x).fit().params; return (f_[0], f_[1]) + sd.expb_to_depth_params(f_[0], f_[1])
idx = rng.integers(0, len(cal), (NBOOT, len(cal))); bs = np.array([_bp(cal.SC30.values[i], cal.SOC100.values[i]) for i in idx])
ci = np.percentile(bs, [2.5, 97.5], axis=0)
out["linear"].update(k_per_cm=kk, Cb_Mg_ha_cm=cb, halving_depth_cm=np.log(2) / kk, boot_ci=dict(a=list(ci[:, 0]), F=list(ci[:, 1]), k=list(ci[:, 2]), Cb=list(ci[:, 3])))
out["loglog"] = dict(a=g1.params["Intercept"], b=g1.params["lx"], b_lo=g1.conf_int().loc["lx", 0], b_hi=g1.conf_int().loc["lx", 1],
                     p_b_eq_1=float(g1.t_test("lx = 1").pvalue), R2=g1.rsquared, smear=float(np.mean(np.exp(g1.resid))),
                     BP_p=het_breuschpagan(g1.resid, g1.model.exog)[1], shapiro_p=stats.shapiro(g1.resid).pvalue)
out["spearman_ratio_vs_SOC30"] = dict(zip(["rho", "p"], map(float, stats.spearmanr(cal.SC30, cal.ratio))))

# =============================================================== 3. cross-validation
def run_cv(splits):
    P = {m: np.full(len(cal), np.nan) for m in sd.MODELS}
    for tr_i, te_i in splits:
        a, b = cal.iloc[tr_i], cal.iloc[te_i]
        for m, (_, _, fn) in sd.MODELS.items(): P[m][te_i] = fn(a)(b)
    return pd.DataFrame(P)

obs = cal.SOC100.values
# (i) repeated random k-fold
cvP = []; cvM = []
for r in range(NREP):
    kf = KFold(NFOLD, shuffle=True, random_state=SEED + r); P = run_cv(list(kf.split(cal))); cvP.append(P)
    for m in sd.MODELS: cvM.append(dict(model=m, rep=r, **sd.metrics(obs, P[m].values)))
cvM = pd.DataFrame(cvM); cv_mean = sum(cvP) / NREP
# (ii) leave-one-source-out
src = cal.Source.values; loso_splits = [(np.where(src != s)[0], np.where(src == s)[0]) for s in np.unique(src)]
losoP = run_cv(loso_splits)
# (iii) spatial block CV (k-means on projected coordinates)
blk = KMeans(10, n_init=10, random_state=SEED).fit_predict(cal[["E", "N"]].values); cal["block"] = blk
sbP = run_cv([(np.where(blk != b)[0], np.where(blk == b)[0]) for b in np.unique(blk)])

def table(P, name):
    rows = []
    for m, (lab, fam, _) in sd.MODELS.items():
        e = P[m].values - obs
        rows.append(dict(model=m, label=lab, family=fam, **sd.metrics(obs, P[m].values),
                         bias_p=stats.wilcoxon(e).pvalue, bias_t_p=stats.ttest_1samp(e, 0).pvalue))
    T = pd.DataFrame(rows); T["bias_p_holm"] = multipletests(T.bias_p, method="holm")[1]
    best = T.loc[T.RMSE.idxmin(), "model"]; ae_b = np.abs(P[best].values - obs)
    T["p_vs_best"] = [np.nan if m == best else stats.wilcoxon(np.abs(P[m].values - obs), ae_b).pvalue for m in T.model]
    T["p_vs_final"] = [np.nan if m == FINAL else stats.wilcoxon(np.abs(P[m].values - obs), np.abs(P[FINAL].values - obs)).pvalue for m in T.model]
    for c_ in ["p_vs_best", "p_vs_final"]:
        ok = T[c_].notna(); T.loc[ok, c_ + "_holm"] = multipletests(T.loc[ok, c_], method="holm")[1]
    T.to_csv(TAB / name, index=False); return T
T3a = table(cv_mean, "T3a_cv_random.csv")
sdv = cvM.groupby("model")[["ME", "MAE", "RMSE", "R2", "CCC"]].std().add_suffix("_sd_reps"); T3a = T3a.merge(sdv, left_on="model", right_index=True)
T3a.to_csv(TAB / "T3a_cv_random.csv", index=False)
T3b = table(losoP, "T3b_cv_leave_source_out.csv"); T3c = table(sbP, "T3c_cv_spatial_block.csv")
cols = ["model", "ME", "PBIAS", "MAE", "RMSE", "MdAPE", "R2", "CCC", "bias_p_holm", "p_vs_final_holm"]
for nm, T in [("RANDOM 10x10 CV", T3a), ("LEAVE-ONE-SOURCE-OUT", T3b), ("SPATIAL BLOCK", T3c)]:
    print("\n" + nm); print(T[cols].round(3).to_string())

# per-source LOSO, per-LULC and per-SOC30-tercile bias
cal["tercile"] = pd.qcut(cal.SC30, 3, labels=["low", "mid", "high"])
out["tercile_edges"] = [float(v) for v in np.percentile(cal.SC30, [0, 33.333, 66.667, 100])]
def strat(P, by, name):
    rows = []
    for g, idx in cal.groupby(by, observed=True).indices.items():
        for m in sd.MODELS:
            mt = sd.metrics(obs[idx], P[m].values[idx]); e = P[m].values[idx] - obs[idx]
            rows.append(dict(group=g, model=m, **mt, bias_p=stats.wilcoxon(e).pvalue if len(idx) > 5 else np.nan))
    pd.DataFrame(rows).to_csv(TAB / name, index=False)
strat(losoP, "Source", "T_S4_loso_by_source.csv"); strat(cv_mean, "LULC", "T_S5_cv_by_lulc.csv"); strat(losoP, "LULC", "T_S5b_loso_by_lulc.csv")
strat(cv_mean, "tercile", "T_S6_cv_by_soc30_tercile.csv"); strat(losoP, "tercile", "T_S6b_loso_by_soc30_tercile.csv")
pd.concat([cal[["num", "Source", "LULC", "LULC_detail", "KG", "SC30", "SOC100", "ratio", "block", "tercile"]], cv_mean.add_prefix("cv_"), losoP.add_prefix("loso_"),
           sbP.add_prefix("sb_")], axis=1).to_csv(PROC / "calibration_with_cv_predictions.csv", index=False)

# back-transformation bias and ratio-estimator variants (random CV, one repetition)
kf = KFold(NFOLD, shuffle=True, random_state=SEED); ex = {k_: np.full(len(cal), np.nan) for k_ in ["LL-naive", "R-meanofratios", "R-median", "R-ratioofmeans", "R-LS"]}
for tr_i, te_i in kf.split(cal):
    a, b = cal.iloc[tr_i], cal.iloc[te_i]
    ex["LL-naive"][te_i] = sd.m_loglog_naive(a)(b); ex["R-meanofratios"][te_i] = a.ratio.mean() * b.SC30
    ex["R-median"][te_i] = a.ratio.median() * b.SC30; ex["R-ratioofmeans"][te_i] = a.SOC100.sum() / a.SC30.sum() * b.SC30
    ex["R-LS"][te_i] = sd.ls_ratio(a.SC30.values, a.SOC100.values) * b.SC30
pd.DataFrame([dict(variant=k_, **sd.metrics(obs, v), bias_p=stats.wilcoxon(v - obs).pvalue) for k_, v in ex.items()]).to_csv(TAB / "T_S7_estimator_variants.csv", index=False)

# sensitivity to QC decisions: refit pooled ratio with excluded pairs added back
allp = df[(df.has100) & (df.LULC != "Mangroves")]
sens = []
for nm, s in [("main (QC ok)", cal), ("+ SOC100 = SOC30 pairs", allp[allp.QC.isin(["ok", "SOC100_equals_SOC30"])]),
              ("+ PERR-FH", allp[allp.QC.isin(["ok", "anomalous_source"])]), ("all upland pairs", allp),
              ("main, ratio <= 3.33 only", cal[cal.ratio <= sd.UNIFORM_R])]:
    R = sd.wls_ratio(s.SC30.values, s.SOC100.values); aF = sd.m_expb_pool(s).params
    sens.append(dict(set=nm, n=len(s), R_fit=R, beta=sd.beta_from_ratio([R])[0], ratio_median=s.ratio.median(), expb_a=aF[0], expb_F=aF[1]))
pd.DataFrame(sens).to_csv(TAB / "T_S8_qc_sensitivity.csv", index=False)

# mangrove contrast case
mg = df[(df.role == "mangrove_paired") & (df.QC == "ok")]
rowsm = [dict(model=m, **sd.metrics(mg.SOC100.values, sd.MODELS[m][2](cal)(mg))) for m in ["UNI", "B-JJ", "R-pool"]]
Rm = sd.wls_ratio(mg.SC30.values, mg.SOC100.values)
out["mangrove"] = dict(n_all=int((df.role == "mangrove_paired").sum()), n_ok=len(mg), ratio_median=mg.ratio.median(), ratio_q1=mg.ratio.quantile(.25), ratio_q3=mg.ratio.quantile(.75),
                       R_fit=Rm, pct_ratio_gt_uniform=100 * (mg.ratio > sd.UNIFORM_R).mean(), p_vs_uniform=stats.wilcoxon(mg.ratio - sd.UNIFORM_R).pvalue,
                       spearman=dict(zip(["rho", "p"], map(float, stats.spearmanr(mg.SC30, mg.ratio)))))
pd.DataFrame(rowsm).to_csv(TAB / "T_S9_mangrove_contrast.csv", index=False)

# =============================================================== 4. final model, prediction intervals, harmonised dataset
fin = sd.MODELS[FINAL][2](cal); pool = sd.m_ratio_pool(cal); rl = sd.m_ratio_lulc(cal);  ll = sd.m_loglog_pool(cal)
coef = pd.DataFrame([dict(LULC=l, n_cal=int((cal.LULC == l).sum()), R_LULC=rl.R.get(l, rl.Rpool)) for l in sd.LULC_ORDER] +
                    [dict(LULC="Pooled", n_cal=len(cal), R_LULC=pool.R)])
coef["beta"] = sd.beta_from_ratio(coef.R_LULC.values); coef["k_per_cm"] = -np.log(coef.beta); coef.to_csv(TAB / "T4_proportional_coefficients.csv", index=False)
print(coef.round(4).to_string())
out["final_model"] = dict(id=FINAL, a=fin.params[0], F=fin.params[1], loglog=dict(a=ll.params[0], b=ll.params[1], smear=ll.params[2]), R_pool=pool.R, beta_pool=float(sd.beta_from_ratio([pool.R])[0]))
# empirical 90 % prediction intervals: quantiles of out-of-sample obs/pred, by SOC30 tercile (pooled over repeated CV and LOSO)
edges = np.percentile(cal.SC30, [33.333, 66.667]); tc = np.digitize(cal.SC30.values, edges)
pi = {}
for g in range(3):
    m_ = tc == g; rel = np.concatenate([obs[m_] / P[FINAL].values[m_] for P in cvP] + [obs[m_] / losoP[FINAL].values[m_]] * NREP)
    pi[g] = np.percentile(rel, [5, 95])
def interval(x, p): g = np.digitize(x, edges); return p * np.array([pi[i][0] for i in g]), p * np.array([pi[i][1] for i in g])
cover = {}
for k_, P in [("cv", cv_mean[FINAL].values), ("loso", losoP[FINAL].values), ("spatial", sbP[FINAL].values)]:
    lo, hi = interval(cal.SC30.values, P); cover[k_] = float(((obs >= lo) & (obs <= hi)).mean())
out["PI"] = dict(tercile_edges=list(edges), factors={["low", "mid", "high"][g]: list(v) for g, v in pi.items()}, coverage=cover)

pred = fin(df); lim = (cal.SC30.min(), cal.SC30.max())
lo, hi = interval(df.SC30.values, pred)
df["SOC100_pred"] = pred; df["SOC100_pred_lo90"] = lo; df["SOC100_pred_hi90"] = hi
df["SOC100_pred_Rpool"] = pool(df); df["SOC100_pred_RLULC"] = rl(df); df["SOC100_pred_betaJJ"] = sd.m_beta_jj(cal)(df)
df["R_implied"] = pred / df.SC30
is_t = df.role == "target"
df["SOC100_harmonised"] = np.where(df.has100, df.SOC100, df.SOC100_pred)
df["SOC100_origin"] = np.where(df.has100, "measured", "predicted_" + FINAL)
df["SOC30_outside_calibration_range"] = (df.SC30 < lim[0]) | (df.SC30 > lim[1])
df["implied_ratio_gt_uniform"] = df.R_implied > sd.UNIFORM_R
pc = ["SOC100_pred", "SOC100_pred_lo90", "SOC100_pred_hi90", "SOC100_pred_Rpool", "SOC100_pred_RLULC", "SOC100_pred_betaJJ", "R_implied"]
df.loc[df.LULC == "Mangroves", pc] = np.nan
df.loc[df.LULC == "Mangroves", ["SOC30_outside_calibration_range", "implied_ratio_gt_uniform"]] = False
keep = ["num", "antsa", "Database", "Source", "Year", "Plot", "Latitude", "Longitude", "KG", "LU", "LULC_detail", "LULC", "LULC_source", "WC2021", "ValFlag", "SC30", "SOC100", "ratio", "QC", "role",
        "SOC100_pred", "SOC100_pred_lo90", "SOC100_pred_hi90", "R_implied", "SOC30_outside_calibration_range", "implied_ratio_gt_uniform",
        "SOC100_pred_Rpool", "SOC100_pred_RLULC", "SOC100_pred_betaJJ", "SOC100_harmonised", "SOC100_origin"]
H = df[keep].rename(columns={"SC30": "SOC30"})
H.to_csv(PROC / "SOC_Madagascar_harmonised_0_100cm.csv", index=False)
meta = pd.DataFrame([
 ("SOC30", "Measured SOC stock 0-30 cm (Mg C ha-1)"), ("SOC100", "Measured SOC stock 0-100 cm (Mg C ha-1); blank if not measured"),
 ("KG", "Koppen-Geiger climate class 1986-2010 (Rubel et al., 2017)"), ("LU / LULC_detail / LULC", "Original land-use label / harmonised detailed class / land-use group used in the analysis"),
 ("LULC_source", "field = from the field label; WorldCover 2021 = from ESA WorldCover 2021 (Zanaga et al., 2022) where the field label was undetermined"),
 ("WC2021", "ESA WorldCover 2021 class code at the plot (10 tree cover, 20 shrubland, 30 grassland, 40 cropland, 50 built-up, 60 bare/sparse, 80 water, 90 herbaceous wetland, 95 mangroves)"), ("ratio", "SOC100 / SOC30 for paired plots"),
 ("QC", "Quality flag of the measured pair (paper, section 2.2)"), ("role", "calibration | target | mangrove_paired | paired_excluded"),
 ("SOC100_pred", f"SOC 0-100 cm predicted by the final model ({FINAL}: SOC100 = a + F x SOC30); given for all non-mangrove plots"),
 ("SOC100_pred_lo90 / hi90", "Empirical 90 % prediction interval from out-of-sample relative errors, by SOC30 tercile"),
 ("R_implied", "SOC100_pred / SOC30"), ("SOC30_outside_calibration_range", "TRUE if SOC30 lies outside the range of the calibration plots (extrapolation)"),
 ("implied_ratio_gt_uniform", "TRUE if the prediction implies C density increasing with depth (ratio > 3.33); occurs only at very low SOC30"),
 ("SOC100_pred_Rpool", "Alternative: pooled fitted beta (proportional) model"), ("SOC100_pred_RLULC", "Alternative: LULC-specific fitted beta model (pooled value where LULC is undetermined)"),
 ("SOC100_pred_betaJJ", "Alternative: a-priori global beta = 0.9786"),
 ("SOC100_harmonised", "Measured SOC100 where available, otherwise SOC100_pred. Measured values with QC != ok are kept as reported: filter on QC as needed"),
 ("SOC100_origin", "measured | predicted")], columns=["column", "description"])
with pd.ExcelWriter(PROC / "SOC_Madagascar_harmonised_0_100cm.xlsx") as xw:
    H.to_excel(xw, sheet_name="harmonised_data", index=False); meta.to_excel(xw, sheet_name="README", index=False)

t = df[is_t]
T5 = t.groupby("LULC").agg(n=("SC30", "size"), SOC30_mean=("SC30", "mean"), SOC100_pred_mean=("SOC100_pred", "mean"), SOC100_pred_median=("SOC100_pred", "median"),
                           betaJJ_mean=("SOC100_pred_betaJJ", "mean"), Rpool_mean=("SOC100_pred_Rpool", "mean"), RLULC_mean=("SOC100_pred_RLULC", "mean"),
                           uniform_mean=("SC30", lambda s: s.mean() * sd.UNIFORM_R), n_outside_range=("SOC30_outside_calibration_range", "sum")).reset_index()
tot = pd.DataFrame([dict(LULC="All target plots", n=len(t), SOC30_mean=t.SC30.mean(), SOC100_pred_mean=t.SOC100_pred.mean(), SOC100_pred_median=t.SOC100_pred.median(),
                         betaJJ_mean=t.SOC100_pred_betaJJ.mean(), Rpool_mean=t.SOC100_pred_Rpool.mean(), RLULC_mean=t.SOC100_pred_RLULC.mean(), uniform_mean=t.SC30.mean() * sd.UNIFORM_R, n_outside_range=t.SOC30_outside_calibration_range.sum())])
T5 = pd.concat([T5, tot]); T5["diff_betaJJ_pct"] = 100 * (T5.betaJJ_mean / T5.SOC100_pred_mean - 1); T5["diff_Rpool_pct"] = 100 * (T5.Rpool_mean / T5.SOC100_pred_mean - 1)
T5["diff_RLULC_pct"] = 100 * (T5.RLULC_mean / T5.SOC100_pred_mean - 1); T5["diff_uniform_pct"] = 100 * (T5.uniform_mean / T5.SOC100_pred_mean - 1)
T5.to_csv(TAB / "T5_predictions_target_plots.csv", index=False); print(T5.round(1).to_string())
out["flags"] = dict(outside_range=int(t.SOC30_outside_calibration_range.sum()), below=int((t.SC30 < lim[0]).sum()), above=int((t.SC30 > lim[1]).sum()), ratio_gt_uniform=int(t.implied_ratio_gt_uniform.sum()), pi_rel_width=float(((t.SOC100_pred_hi90 - t.SOC100_pred_lo90) / t.SOC100_pred).median())); out["n"] = dict(total=len(df), cal=len(cal), target=int(is_t.sum()), paired=int(df.has100.sum()))
out["cal_range"] = lim; out["literature"] = dict(beta_JJ_veg=sd.BETA_JJ_VEG, R_JJ=float(sd.ratio_from_beta(sd.BETA_JJ_GLOBAL)),
     R_JJveg={k_: float(sd.ratio_from_beta(v)) for k_, v in sd.BETA_JJ_VEG.items()}, R_W25={k_: float(sd.ratio_from_beta(v)) for k_, v in sd.BETA_W25.items()})
json.dump(out, open(TAB / "summary_statistics.json", "w"), indent=1, default=float)
print(json.dumps({k_: out[k_] for k_ in ["variance_function", "nested", "linear", "final_model", "PI", "flags", "n", "cal_range"]}, indent=1, default=float))
