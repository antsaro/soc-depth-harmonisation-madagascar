"""Figure 1: study area map (SOC 0-30 cm and 0-100 cm plots, protected areas within 10 km of a plot).

Usage: python src/03_study_area_map.py data/raw/SOCv5.xlsx <WDPA polygons for Madagascar (.shp/.gpkg)>
The protected-area polygons (WDPA, UNEP-WCMC & IUCN) are not redistributed here; download them from protectedplanet.net.
"""
import sys, numpy as np, pandas as pd, geopandas as gpd, pyproj
import matplotlib, matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patheffects as pe

from pathlib import Path
XLSX, PA_SHP = sys.argv[1:3]
ROOT = Path(__file__).resolve().parents[1]; BASE = ROOT / "data/basemap"
OUT = str(ROOT / "outputs/figures/Fig1_study_area"); (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
PA_DIST_KM = 10          # protected areas kept if within this distance of any sampling plot

from matplotlib import font_manager
_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Helvetica", "TeX Gyre Heros", "Liberation Sans") if f in _fonts), "DejaVu Sans")
plt.rcParams.update({
    "font.family": FONT, "font.size": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
})
OCEAN, LAND, COAST = "#e9f1f6", "#f6f3ec", "#8d8a82"
PA_FILL, PA_EDGE = "#bfe3b4", "#2f8f4e"
INK, EDGE = "#1a1a1a", "#14141a"
halo = [pe.withStroke(linewidth=1.8, foreground="white", alpha=0.9)]

CRS = 3395
T = pyproj.Transformer.from_crs(4326, CRS, always_xy=True)
def mx(lon): return T.transform(lon, 0)[0]
def my(lat): return T.transform(0, lat)[1]

# ---------- data ----------
land = gpd.read_file(BASE / "madagascar_coast_gshhs_full.gpkg")
_c = land.geometry.representative_point()
land = land[~((_c.x < 46.0) & (_c.y > -13.6))].to_crs(CRS)       # drop Comoros archipelago
world = gpd.read_file(BASE / "ne_countries_africa.gpkg")
places = gpd.read_file(BASE / "ne_places_madagascar.gpkg")

df = pd.read_excel(XLSX)
pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs=4326)
pa = gpd.read_file(PA_SHP)
pa = pa[pa.ISO3.str.contains("MDG")].to_crs(32738)
pa["geometry"] = pa.buffer(0)
p32 = pts.to_crs(32738)
for col in ("SC30", "SOC100"):                                    # selection is done per panel
    near = p32[p32[col].notna()].buffer(PA_DIST_KM * 1000).union_all()
    pa["near_" + col] = pa.intersects(near)
pa = pa[pa.near_SC30 | pa.near_SOC100].copy()
pa[["NAME", "DESIG_ENG", "IUCN_CAT", "STATUS", "GIS_AREA", "near_SC30", "near_SOC100"]].sort_values("NAME").to_csv(ROOT / "outputs/tables/T_S12_protected_areas_near_plots.csv", index=False)
pa_all = pa.to_crs(CRS)
pts = pts.to_crs(CRS).sample(frac=1, random_state=7)

# vivid sequential ramp (light yellow -> orange -> magenta -> deep purple), 6 classes
ramp = [matplotlib.colormaps["inferno_r"](v) for v in np.linspace(0.04, 0.86, 6)]
ramp[0] = (1.0, 0.93, 0.45, 1.0)
PANELS = [
    ("a", "SOC stock, 0–30 cm",  "SC30",   [25, 50, 75, 100, 150]),
    ("b", "SOC stock, 0–100 cm", "SOC100", [75, 150, 250, 400, 600]),
]

# ---------- layout (mm) ----------
LEG = 9.0
W, PW, X0, GAP, Y0 = 183.0, 82.0, 10.0, 5.0, LEG + 5.5
EXT = (42.7, 50.9, -25.9, -11.6)
ph = PW * (my(EXT[3]) - my(EXT[2])) / (mx(EXT[1]) - mx(EXT[0]))
H = ph + Y0 + 6.5
fig = plt.figure(figsize=(W / 25.4, H / 25.4))

CITIES = {"Antananarivo": (5, 5, "left"), "Antsiranana": (-5, 7, "right"), "Mahajanga": (-4, 8, "right"),
          "Toamasina": (2, -8, "center"), "Toliara": (6, -2, "left"), "Morondava": (-6, 0, "right"),
          "Taolanaro": (6, -5, "left"), "Fianarantsoa": (6, -4, "left")}
pl = places[places.adm0name == "Madagascar"].to_crs(CRS)

for k, (letter, title, col, breaks) in enumerate(PANELS):
    ax = fig.add_axes([(X0 + k * (PW + GAP)) / W, Y0 / H, PW / W, ph / H]); ax.set_facecolor(OCEAN)
    land.plot(ax=ax, color=LAND, edgecolor=COAST, linewidth=0.25, zorder=1)
    pa_m = pa_all[pa_all["near_" + col]]; pa_union = gpd.GeoSeries([pa_m.union_all()], crs=CRS)
    print(letter, "protected areas shown:", len(pa_m))
    pa_union.plot(ax=ax, color=PA_FILL, alpha=0.75, edgecolor="none", zorder=2)
    pa_m.boundary.plot(ax=ax, color=PA_EDGE, linewidth=0.35, zorder=3)
    land.boundary.plot(ax=ax, color=COAST, linewidth=0.25, zorder=3.5)

    sub = pts[pts[col].notna()]
    cls = np.digitize(sub[col].values, breaks)
    ax.scatter(sub.geometry.x, sub.geometry.y, s=11, c=[ramp[i] for i in cls], edgecolors=EDGE,
               linewidths=0.3, zorder=6)
    print(letter, col, "n =", len(sub), "class counts:", np.bincount(cls, minlength=6).tolist())

    ax.set_xlim(mx(EXT[0]), mx(EXT[1])); ax.set_ylim(my(EXT[2]), my(EXT[3]))
    lons = np.arange(44, 51, 2); lats = np.arange(-24, -11, 2)
    ax.set_xticks([mx(v) for v in lons]); ax.set_xticklabels([f"{v:g}° E" for v in lons], fontsize=7.5, color="#3d3d3d")
    ax.set_yticks([my(v) for v in lats])
    ax.set_yticklabels([f"{abs(v):g}° S" for v in lats] if k == 0 else [], fontsize=7.5, color="#3d3d3d",
                       rotation=90, va="center")
    ax.tick_params(colors="#3d3d3d", top=True, right=True); ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values(): s.set_edgecolor("#3a3a3a")

    # cities
    for _, r in pl.iterrows():
        key = next((c for c in CITIES if c.lower() in (str(r["name"]).lower(), str(r.get("nameascii", "")).lower())), None)
        if key is None: continue
        ox, oy, ha = CITIES[key]
        ax.plot(r.geometry.x, r.geometry.y, marker="s", ms=2.8, mfc="white", mec=INK, mew=0.7, zorder=8)
        ax.annotate("Tôlanaro" if key == "Taolanaro" else key, (r.geometry.x, r.geometry.y), xytext=(ox, oy),
                    textcoords="offset points", ha=ha, va="center", fontsize=7.5, color="#2b2b2b", zorder=9,
                    path_effects=halo)
    ax.text(mx(43.45), my(-17.2), "Mozambique Channel", rotation=78, fontsize=9, style="italic", color="#5d86a3",
            ha="center", va="center", zorder=2)
    ax.text(mx(50.2), my(-19.4), "Indian Ocean", rotation=72, fontsize=9, style="italic", color="#5d86a3",
            ha="center", va="center", zorder=2)

    # panel label + title
    ax.text(0, 1, letter, transform=ax.transAxes + matplotlib.transforms.ScaledTranslation(0, 3 / 72, fig.dpi_scale_trans),
            ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)
    ax.text(0, 1, f"{title}  (n = {len(sub):,} plots)",
            transform=ax.transAxes + matplotlib.transforms.ScaledTranslation(12.5 / 72, 3.8 / 72, fig.dpi_scale_trans),
            ha="left", va="bottom", fontsize=9, color=INK)

    # class legend
    b = breaks
    labs = [f"< {b[0]}"] + [f"{b[i]}–{b[i + 1]}" for i in range(len(b) - 1)] + [f"> {b[-1]}"]
    hs = [Line2D([], [], marker="o", ls="", ms=6.5, mfc=ramp[i], mec=EDGE, mew=0.45, label=labs[i]) for i in range(6)]
    lg = ax.legend(handles=hs, title="SOC stock\n(Mg C ha$^{-1}$)", loc="lower left", bbox_to_anchor=(0.70, 0.012),
                   fontsize=8, title_fontsize=8.5, frameon=False, handletextpad=0.2, labelspacing=0.38,
                   borderpad=0, alignment="left")
    lg.get_title().set_fontweight("bold")

    if k == 0:
        # scale bar
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        L = 200e3 / np.cos(np.radians(18.8)); xs = x0 + 0.05 * (x1 - x0); ys = y0 + 0.035 * (y1 - y0)
        ax.plot([xs, xs + L], [ys, ys], color=INK, lw=1.3, solid_capstyle="butt", zorder=9)
        for xx in (xs, xs + L / 2, xs + L):
            ax.plot([xx, xx], [ys, ys + 0.008 * (y1 - y0)], color=INK, lw=0.5, zorder=9)
        ax.text(xs + L / 2, ys + 0.016 * (y1 - y0), "200 km", ha="center", va="bottom", fontsize=7.5, color=INK,
                path_effects=halo, zorder=9)
        # locator
        iw = 27.0; ih = iw * 58 / 62
        axi = fig.add_axes([(X0 + 1.8) / W, (Y0 + ph - ih - 1.8) / H, iw / W, ih / H]); axi.set_facecolor("#f4f8fa")
        world.plot(ax=axi, color="#d9d6cf", edgecolor="white", linewidth=0.15)
        world[world.ADM0_A3 == "MDG"].plot(ax=axi, color="#3a3a3a", edgecolor="#3a3a3a", linewidth=0.2)
        axi.add_patch(Rectangle((EXT[0], EXT[2]), EXT[1] - EXT[0], EXT[3] - EXT[2], fill=False, ec="#c0392b", lw=0.6))
        axi.set_xlim(8, 70); axi.set_ylim(-38, 20); axi.set_aspect(1.0)
        axi.set_xticks([]); axi.set_yticks([]); axi.set_xlabel(""); axi.set_ylabel("")
        axi.text(24, 3, "AFRICA", fontsize=6.5, color="#7d7a73", ha="center", va="center")
        axi.text(58, -30, "Indian\nOcean", fontsize=6, color="#6f8fa6", ha="center", va="center", style="italic",
                 linespacing=0.95)
        for s in axi.spines.values(): s.set_linewidth(0.5); s.set_edgecolor("#3a3a3a")

fig.legend(handles=[Patch(fc=PA_FILL, ec=PA_EDGE, lw=0.6,
                          label=f"Protected areas within {PA_DIST_KM} km of a sampling plot"),
                    Line2D([], [], marker="s", ls="", ms=3.4, mfc="white", mec=INK, mew=0.7, label="Cities")],
           loc="center left", bbox_to_anchor=(X0 / W, (LEG - 5.0) / H), ncol=2, fontsize=8.5, frameon=False,
           handlelength=1.6, handletextpad=0.5, columnspacing=2.5, borderpad=0, borderaxespad=0)

print("fig mm:", W, round(H, 1))
# plots inside a protected area (designated or proposed)
inside = gpd.sjoin(gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs=4326).to_crs(32738),
                   gpd.read_file(PA_SHP).pipe(lambda g: g[g.ISO3.str.contains("MDG")]).to_crs(32738)[["geometry"]], predicate="within")
print("plots inside a protected area:", inside.index.nunique())
fig.savefig(f"{OUT}.png", dpi=600); fig.savefig(f"{OUT}.pdf")
