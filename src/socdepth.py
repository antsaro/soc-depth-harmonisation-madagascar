"""Depth-harmonisation models for converting SOC stocks 0-30 cm to 0-100 cm.

All stocks in Mg C ha-1. Every model exposes fit(train_df) -> predictor(df) -> np.ndarray.
"""
import warnings
import numpy as np, pandas as pd
from scipy.optimize import brentq
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor

D0, D1 = 30.0, 100.0
UNIFORM_R = D1 / D0

# ---------------------------------------------------------------- land use
LULC_MAP = {
    "Forest": "Forests", "Closed Canopy": "Forests",
    "Grassland": "Grasslands", "Pseudo-steppe": "Grasslands",
    "Shrubland": "Shrublands",
    "Reforestation": "Tree plantations", "Pine": "Tree plantations",
    "AgroF": "Agroforestry systems", "AGroF": "Agroforestry systems", "Agroforesterie": "Agroforestry systems",
    "Crop": "Croplands", "Cropland": "Croplands", "Culture": "Croplands",
    "Rice": "Rice fields",
    "Tree fallow": "Fallows", "Shrub fallow": "Fallows", "Jachere": "Fallows",
    "Tany Maty": "Degraded lands", "Tany maty": "Degraded lands", "Bare": "Degraded lands",
    "Mangroves": "Mangroves", "undetermined": "Undetermined",
}
KG_ORDER = ["Af", "Am", "Aw", "BSh", "Cfa", "Cfb", "Cwa", "Cwb"]
KG_NAMES = {"Af": "Tropical rainforest", "Am": "Tropical monsoon", "Aw": "Tropical savanna", "BSh": "Hot semi-arid",
            "Cfa": "Humid subtropical", "Cfb": "Temperate oceanic", "Cwa": "Subtropical, dry winter", "Cwb": "Subtropical highland, dry winter"}

def koppen(lat, lon):
    """Koppen-Geiger class 1986-2010 (Rubel et al., 2017; 100 arc-second grid, via kgcpy); nearest land pixel for coastal plots."""
    from kgcpy import lookupCZ, img, kg_zoneNum_df
    z = lookupCZ(lat, lon)
    if z != "Ocean": return z
    x = round((lon + 180) * img.size[0] / 360 - 0.5); y = round(-(lat - 90) * img.size[1] / 180 - 0.5)
    lut = dict(zip(kg_zoneNum_df.zoneNum, kg_zoneNum_df.kg_zone))
    for r in range(1, 10):
        cand = sorted((dx * dx + dy * dy, lut.get(img.getpixel((x + dx, y + dy)))) for dx in range(-r, r + 1) for dy in range(-r, r + 1))
        cand = [c for c in cand if c[1] not in (None, "Ocean")]
        if cand: return cand[0][1]
    return None

# LULC_MAP gives the detailed class; LULC_GROUP merges detailed classes into the groups used for modelling
LULC_DETAIL_ORDER = ["Forests", "Tree plantations", "Shrublands", "Grasslands", "Degraded lands",
                     "Croplands", "Rice fields", "Agroforestry systems", "Fallows"]
LULC_GROUP = {"Forests": "Forests and tree plantations", "Tree plantations": "Forests and tree plantations",
              "Shrublands": "Shrublands",
              "Grasslands": "Grasslands and degraded lands", "Degraded lands": "Grasslands and degraded lands",
              "Croplands": "Agroecosystems", "Rice fields": "Agroecosystems", "Agroforestry systems": "Agroecosystems", "Fallows": "Agroecosystems",
              "Mangroves": "Mangroves", "Undetermined": "Undetermined"}
LULC_ORDER = ["Forests and tree plantations", "Shrublands", "Grasslands and degraded lands", "Agroecosystems"]

# ESA WorldCover 2021 v200 (Zanaga et al., 2022): used only where the field land use is undetermined
WC_NAMES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse vegetation",
            80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen"}
WC_GROUP = {10: "Forests and tree plantations", 20: "Shrublands", 30: "Grasslands and degraded lands", 60: "Grasslands and degraded lands",
            40: "Agroecosystems", 90: "Agroecosystems", 95: "Mangroves"}        # built-up and water -> Undetermined
WC_COL = "WC2021"                                                               # class of the 10 m pixel at the plot location

# ---------------------------------------------------------------- beta model (Jobbagy & Jackson 2000)
def ratio_from_beta(beta, d0=D0, d1=D1):
    """Cumulative-stock ratio SOC(0-d1)/SOC(0-d0) implied by Y = 1 - beta**d."""
    beta = np.asarray(beta, float)
    return (1 - beta ** d1) / (1 - beta ** d0)

def beta_from_ratio(r, d0=D0, d1=D1):
    """Invert ratio_from_beta; defined for 1 < r < d1/d0."""
    r = np.atleast_1d(np.asarray(r, float)); out = np.full(r.shape, np.nan)
    for i, ri in enumerate(r):
        if 1.0 + 1e-9 < ri < d1 / d0 - 1e-9:
            out[i] = brentq(lambda b: ratio_from_beta(b, d0, d1) - ri, 1e-6, 1 - 1e-9)
    return out

def beta_from_top_fraction(frac, dtop=20.0, dref=100.0):
    """beta such that SOC(0-dtop)/SOC(0-dref) = frac (Jobbagy & Jackson report frac for dtop=20, dref=100)."""
    return brentq(lambda b: (1 - b ** dtop) / (1 - b ** dref) - frac, 1e-6, 1 - 1e-9)

# A-priori (literature) beta hypotheses ------------------------------------------------
BETA_JJ_GLOBAL = 0.9786                      # global average, Jobbagy & Jackson (2000) as used by Yang et al. (2011), Deng et al. (2014)
JJ_TOP20 = {"forest": 0.50, "grassland": 0.42, "shrubland": 0.33}   # share of 0-100 cm SOC found in 0-20 cm (J&J 2000)
BETA_JJ_VEG = {k: beta_from_top_fraction(v) for k, v in JJ_TOP20.items()}
JJ_CLASS = {"Forests and tree plantations": "forest", "Shrublands": "shrubland", "Grasslands and degraded lands": "grassland",
            "Agroecosystems": None, "Undetermined": None}                          # None -> global beta
BETA_W25 = {"cropland": 0.9731, "grassland": 0.9772, "forestland": 0.9790}   # Wang et al. (2025), field observations
W25_CLASS = {"Forests and tree plantations": "forestland", "Shrublands": "grassland", "Grasslands and degraded lands": "grassland",
             "Agroecosystems": "cropland", "Undetermined": None}

def lit_ratio(lulc, scheme):
    if scheme == "JJveg":
        k = JJ_CLASS.get(lulc); b = BETA_JJ_VEG[k] if k else BETA_JJ_GLOBAL
    elif scheme == "W25":
        k = W25_CLASS.get(lulc); b = BETA_W25[k] if k else BETA_JJ_GLOBAL
    else:
        b = BETA_JJ_GLOBAL
    return float(ratio_from_beta(b))

# ---------------------------------------------------------------- models
def m_uniform(train):
    return lambda d: UNIFORM_R * d.SC30.values
def m_beta_jj(train):
    R = float(ratio_from_beta(BETA_JJ_GLOBAL)); return lambda d: R * d.SC30.values
def m_beta_jjveg(train):
    return lambda d: d.LULC.map(lambda l: lit_ratio(l, "JJveg")).values * d.SC30.values
def m_beta_w25(train):
    return lambda d: d.LULC.map(lambda l: lit_ratio(l, "W25")).values * d.SC30.values

def ls_ratio(x, y):
    """Ordinary least squares through the origin (assumes constant error variance)."""
    return float(np.sum(x * y) / np.sum(x * x))

def wls_ratio(x, y):
    """Weighted least squares through the origin with Var(e) proportional to x = ratio of means (Cochran, 1977)."""
    return float(np.sum(y) / np.sum(x))

def m_ratio_pool(train):
    """Proportional model = pooled fitted beta / exponential (any one-parameter depth function); WLS, Var ~ SOC30."""
    R = wls_ratio(train.SC30.values, train.SOC100.values); f = lambda d: R * d.SC30.values; f.R = R; return f

def _ratio_group(train, col):
    """Group-specific proportional model with partial pooling (random slopes); = fitted beta per group.
    Fitted on the variance-stabilised scale y/sqrt(x) = R*sqrt(x) + e. Unseen groups get the pooled ratio."""
    Rp = wls_ratio(train.SC30.values, train.SOC100.values); sx = np.sqrt(train.SC30.values)[:, None]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = sm.MixedLM(train.SOC100.values / sx[:, 0], sx, groups=train[col].values, exog_re=sx).fit(reml=True, method="lbfgs")
        fe = float(md.fe_params[0]); Rg = {g: fe + float(np.asarray(v)[0]) for g, v in md.random_effects.items()}
        if not np.all(np.isfinite(list(Rg.values()))): raise ValueError
        Rp = fe
    except Exception:
        Rg = {}
        for g, s_ in train.groupby(col):
            n = len(s_); Rg[g] = (n * wls_ratio(s_.SC30.values, s_.SOC100.values) + 10 * Rp) / (n + 10)
    f = lambda d: d[col].map(lambda l: Rg.get(l, Rp)).values * d.SC30.values; f.R = Rg; f.Rpool = Rp; return f

def m_ratio_lulc(train): return _ratio_group(train, "LULC")

def m_expb_pool(train):
    """Exponential decline towards a non-zero background density: SOC100 = a + F*SOC30 (WLS, Var ~ SOC30)."""
    X = sm.add_constant(train.SC30.values); fit = sm.WLS(train.SOC100.values, X, weights=1 / train.SC30.values).fit(); a, F = fit.params
    f = lambda d: a + F * d.SC30.values; f.params = (float(a), float(F)); return f

def _expb_group(train, col):
    """As m_expb_pool with group-specific intercepts (background term) and a common slope; unseen groups -> pooled."""
    lv = sorted(train[col].unique()); t = train.assign(G=pd.Categorical(train[col], lv))
    fit = smf.wls("SOC100 ~ SC30 + C(G)", t, weights=1 / t.SC30).fit(); pool = m_expb_pool(train)
    def f(d):
        out = pool(d); ok = d[col].isin(lv).values
        if ok.any():
            out[ok] = fit.predict(d[ok].assign(G=pd.Categorical(d[col][ok], lv))).values
        return out
    return f

def m_expb_lulc(train): return _expb_group(train, "LULC")

def expb_to_depth_params(a, F):
    """Translate (a, F) into decay constant k (cm-1) and background density Cb (Mg C ha-1 cm-1)."""
    k = -np.log(beta_from_ratio([F])[0]); return float(k), float(a / (D1 - D0 * F))

def _smear(resid): return float(np.mean(np.exp(resid)))      # Duan (1983) smearing factor

def m_loglog_pool(train):
    """Allometric model ln SOC100 = a + b ln SOC30, back-transformed with Duan's smearing estimator."""
    X = sm.add_constant(np.log(train.SC30.values)); fit = sm.OLS(np.log(train.SOC100.values), X).fit()
    a, b = fit.params; s = _smear(fit.resid)
    f = lambda d: s * np.exp(a + b * np.log(d.SC30.values)); f.params = (a, b, s); return f

def m_loglog_naive(train):
    """Same as m_loglog_pool but WITHOUT bias correction (shown to illustrate back-transformation bias)."""
    X = sm.add_constant(np.log(train.SC30.values)); fit = sm.OLS(np.log(train.SOC100.values), X).fit(); a, b = fit.params
    return lambda d: np.exp(a + b * np.log(d.SC30.values))

def m_loglog_lulc(train):
    lv = sorted(train.LULC.unique()); t = train.assign(LULC=pd.Categorical(train.LULC, lv), lx=np.log(train.SC30), ly=np.log(train.SOC100))
    fit = smf.ols("ly ~ lx + C(LULC)", t).fit(); s = _smear(fit.resid); pool = m_loglog_pool(train)
    def f(d):
        out = pool(d); ok = d.LULC.isin(lv).values
        if ok.any():
            dd = d[ok].assign(LULC=pd.Categorical(d.LULC[ok], lv), lx=np.log(d.SC30[ok]))
            out[ok] = s * np.exp(fit.predict(dd).values)
        return out
    return f

def m_rf(train):
    """Random-forest benchmark using SOC30, LULC and coordinates."""
    def X(d):
        Z = pd.DataFrame({"SC30": d.SC30.values, "lat": d.Latitude.values, "lon": d.Longitude.values})
        for l in LULC_ORDER: Z["lu_" + l] = (d.LULC.values == l).astype(float)
        return Z.values
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(X(train), train.SOC100.values)
    return lambda d: rf.predict(X(d))

MODELS = {   # id: (label, family, function)
    "UNI":      ("Uniform density (R = 3.33)",                 "A priori",   m_uniform),
    "B-JJ":     ("β model, global β = 0.9786",                 "A priori",   m_beta_jj),
    "B-JJveg":  ("β model, vegetation-specific β (J&J 2000)",  "A priori",   m_beta_jjveg),
    "B-W25":    ("β model, ecosystem β (Wang et al. 2025)",    "A priori",   m_beta_w25),
    "R-pool":   ("Fitted β / exponential, pooled",             "Calibrated, proportional", m_ratio_pool),
    "R-LULC":   ("Fitted β / exponential, by LULC",            "Calibrated, proportional", m_ratio_lulc),
    "EXPb-pool": ("Exponential + background, pooled",          "Calibrated, non-proportional", m_expb_pool),
    "EXPb-LULC": ("Exponential + background, by LULC",         "Calibrated, non-proportional", m_expb_lulc),
    "LL-pool":  ("Log–log allometric, pooled",                 "Calibrated, non-proportional", m_loglog_pool),
    "LL-LULC":  ("Log–log allometric, by LULC",                "Calibrated, non-proportional", m_loglog_lulc),
    "RF":       ("Random forest (SOC30, LULC, coordinates)",   "Machine learning", m_rf),
}

# ---------------------------------------------------------------- metrics
def lin_ccc(o, p):
    o, p = np.asarray(o), np.asarray(p)
    return float(2 * np.cov(o, p, bias=True)[0, 1] / (o.var() + p.var() + (o.mean() - p.mean()) ** 2))

def metrics(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float); e = p - o
    return dict(n=len(o), ME=e.mean(), PBIAS=100 * e.sum() / o.sum(), MAE=np.abs(e).mean(),
                RMSE=np.sqrt((e ** 2).mean()), MdAPE=100 * np.median(np.abs(e) / o),
                R2=1 - (e ** 2).sum() / ((o - o.mean()) ** 2).sum(), CCC=lin_ccc(o, p))
