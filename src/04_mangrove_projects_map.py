"""Study area map of the mangrove sampling plots, coloured by project, over the 2020 mangrove extent of Madagascar.

Usage:  python src/04_mangrove_projects_map.py                      (uses the paths below)
        python src/04_mangrove_projects_map.py <plots.xlsx> <gmw_v3_2020_Mada.gpkg>   (other locations)
Output: outputs/figures/Map_mangrove_projects.png (600 dpi) and .pdf
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib, matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from shapely.geometry import box

# ---------- input files: edit these two paths if your files are elsewhere ----------
XLSX = Path(r"C:\Users\User\Desktop\LRI\Data_Mangroves_All_Projects_filled_WBconverted_v2.xlsx")
GPKG = XLSX.parent / "gmw_v3_2020_Mada.gpkg"          # Global Mangrove Watch v3.0, mangrove extent 2020, Madagascar
if len(sys.argv) > 2: XLSX, GPKG = Path(sys.argv[1]), Path(sys.argv[2])
for f in (XLSX, GPKG):
    if not f.exists(): sys.exit(f"File not found: {f}")

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "basemap"                      # coastline (GSHHG) and Natural Earth layers, shipped with the repository
OUT = ROOT / "outputs" / "figures" / "Map_mangrove_projects"; OUT.parent.mkdir(parents=True, exist_ok=True)
from matplotlib import font_manager
_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Helvetica", "TeX Gyre Heros", "Liberation Sans") if f in _fonts), "DejaVu Sans")

# ---------- style (Nature: Helvetica/Arial, 5-7 pt, 183 mm wide) ----------
plt.rcParams.update({
    "font.family": FONT, "font.size": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
})
OCEAN, LAND, COAST = "#e9f1f6", "#f6f3ec", "#8d8a82"
MANG = "#14804a"
INK, MUTED = "#1a1a1a", "#5f6368"
PROJ_COL = {"BV": "#2563ff", "BNCCREDD": "#ff9500", "4MPSR": "#ff3d9a",
            "PEER": "#4b1196", "WWF": "#00c2e0", "USFS_SM": "#ffd21f"}
EDGE = "#14141a"
def nice(p): return p.replace("_", " ")
halo = [pe.withStroke(linewidth=1.8, foreground="white", alpha=0.9)]

# ---------- projection helpers (World Mercator, EPSG:3395) ----------
CRS = 3395
import pyproj
T = pyproj.Transformer.from_crs(4326, CRS, always_xy=True)
def mx(lon): return T.transform(lon, 0)[0]
def my(lat): return T.transform(0, lat)[1]
def fmt_lon(v): return f"{v:g}° E"
def fmt_lat(v): return f"{abs(v):g}° S"

# ---------- data ----------
mang = gpd.read_file(GPKG).to_crs(CRS)
land = gpd.read_file(BASE / "madagascar_coast_gshhs_full.gpkg")
_c = land.geometry.representative_point()
land = land[~((_c.x < 46.0) & (_c.y > -13.6))].to_crs(CRS)   # drop Comoros archipelago (outside study area)
riv = gpd.read_file(BASE / "ne_rivers_madagascar.gpkg").to_crs(CRS)
world = gpd.read_file(BASE / "ne_countries_africa.gpkg")
places = gpd.read_file(BASE / "ne_places_madagascar.gpkg")
df = pd.read_excel(XLSX)
pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=4326).to_crs(CRS)
pts = pts.sample(frac=1, random_state=3)          # shuffle so no project systematically overplots
counts = df.Projet.value_counts()
assert set(counts.index) == set(PROJ_COL), counts
riv_in = gpd.clip(riv, land.union_all())

# ---------- layout (mm) ----------
LEG = 17.0
W, MAIN_W, X0, Y0 = 183.0, 82.0, 10.0, LEG + 5.5
EXT = (42.7, 50.9, -25.9, -11.6)
main_h = MAIN_W * (my(EXT[3]) - my(EXT[2])) / (mx(EXT[1]) - mx(EXT[0]))
H = main_h + Y0 + 6.5
RX, RW, GAPX, GAPY = 102.0, 79.0, 10.0, 11.0
h3 = 52.0; h2 = 36.0; h1 = main_h - h3 - h2 - 2 * GAPY
y3 = Y0; y2 = y3 + h3 + GAPY; y1 = y2 + h2 + GAPY
# letter, title, centre lon/lat, lon span, box(x,y,w,h), scalebar km, tick step
ZOOMS = [
    ("b", "Ambaro–Ambanja bays",              48.68, -13.25, 1.02, (RX, y1, 42, h1),            20, 0.5),
    ("c", "Île Sainte-Marie",                 49.868, -17.01, 0.20, (RX + 42 + GAPX, y1, 27, h1), 5, 0.1),
    ("d", "Mahajamba–Bombetoka–Baly bays",    46.00, -15.80, 2.74, (RX, y2, RW, h2),            50, 1.0),
    ("e", "Menabe coast",                     44.20, -19.79, 1.24, (RX, y3, 26, h3),            50, 0.5),
    ("f", "Baie des Assassins"              , 43.289, -22.208, 0.16, (RX + 26 + GAPX, y3, 43, h3), 3, 0.05),
]
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
def add_ax(x, y, w, h):
    ax = fig.add_axes([x / W, y / H, w / W, h / H]); ax.set_facecolor(OCEAN); return ax

def base_layers(ax, zoom):
    land.plot(ax=ax, color=LAND, edgecolor=COAST, linewidth=0.3 if zoom else 0.25, zorder=1)
    if not zoom:
        riv_in.plot(ax=ax, color="#8fb8d4", linewidth=0.4, zorder=2)
    mang.plot(ax=ax, color=MANG, edgecolor=MANG, linewidth=0.1 if zoom else 0.45, zorder=3)

def draw_points(ax, size):
    ax.scatter(pts.geometry.x, pts.geometry.y, s=size, c=pts.Projet.map(PROJ_COL),
               edgecolors=EDGE, linewidths=0.4, zorder=6, clip_on=True)

def graticule(ax, lon0, lon1, lat0, lat1, step, lstep=None):
    lstep = lstep or step
    lons = np.arange(np.ceil(lon0 / step - 1e-9) * step, lon1 + 1e-9, step).round(6)
    lats = np.arange(np.ceil(lat0 / lstep - 1e-9) * lstep, lat1 + 1e-9, lstep).round(6)
    ax.set_xticks([mx(v) for v in lons]); ax.set_xticklabels([fmt_lon(v) for v in lons], fontsize=7.5, color="#3d3d3d")
    ax.set_yticks([my(v) for v in lats]); ax.set_yticklabels([fmt_lat(v) for v in lats], fontsize=7.5, color="#3d3d3d",
                                                             rotation=90, va="center")
    ax.tick_params(colors=MUTED, top=True, right=True, direction="out")
    for s in ax.spines.values(): s.set_edgecolor("#3a3a3a")

def scalebar(ax, km, latc, loc="lr"):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    L = km * 1000 / np.cos(np.radians(latc))
    dx, dy = 0.05 * (x1 - x0), 0.035 * (y1 - y0)
    xs = x1 - dx - L if loc == "lr" else x0 + dx
    ys = y0 + dy
    ax.plot([xs, xs + L], [ys, ys], color=INK, lw=1.3, solid_capstyle="butt", zorder=9)
    for xx in (xs, xs + L / 2, xs + L):
        ax.plot([xx, xx], [ys, ys + 0.008 * (y1 - y0)], color=INK, lw=0.5, zorder=9)
    ax.text(xs + L / 2, ys + 0.016 * (y1 - y0), f"{km} km", ha="center", va="bottom", fontsize=7.5,
            color=INK, zorder=9, path_effects=halo)

def panel_label(ax, letter, title=None):
    ax.text(0.0, 1.0, letter, transform=ax.transAxes + matplotlib.transforms.ScaledTranslation(0, 3.0 / 72, fig.dpi_scale_trans),
            ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)
    if title:
        ax.text(0.0, 1.0, title, transform=ax.transAxes + matplotlib.transforms.ScaledTranslation(12.5 / 72, 3.8 / 72, fig.dpi_scale_trans),
                ha="left", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel(""); ax.set_ylabel("")

# ================= panel a =================
ax = add_ax(X0, Y0, MAIN_W, main_h)
base_layers(ax, zoom=False)
draw_points(ax, 12)
ax.set_xlim(mx(EXT[0]), mx(EXT[1])); ax.set_ylim(my(EXT[2]), my(EXT[3]))
graticule(ax, *EXT, 2.0)
scalebar(ax, 200, -18.8, loc="ll")
panel_label(ax, "a", "Mangrove extent and sampling plots, Madagascar")

# cities for orientation
CITIES = {"Antananarivo": (5, 0, "left"), "Antsiranana": (-5, 7, "right"), "Mahajanga": (-4, 9, "right"),
          "Toamasina": (-5, 0, "right"), "Toliara": (6, -2, "left"), "Morondava": (24, -4, "left"),
          "Taolanaro": (-5, 4, "right")}
pl = places[places.adm0name == "Madagascar"].to_crs(CRS)
for _, r in pl.iterrows():
    key = next((k for k in CITIES if k.lower() in (str(r["name"]).lower(), str(r.get("nameascii", "")).lower())), None)
    if key is None: continue
    ox, oy, ha = CITIES[key]
    ax.plot(r.geometry.x, r.geometry.y, marker="s", ms=2.6, mfc="white", mec=INK, mew=0.6, zorder=5)
    ax.annotate("Tôlanaro" if key == "Taolanaro" else key, (r.geometry.x, r.geometry.y), xytext=(ox, oy),
                textcoords="offset points", ha=ha, va="center", fontsize=7.5, color="#2b2b2b", zorder=5,
                path_effects=halo)
ax.text(mx(43.5), my(-17.9), "Mozambique Channel", rotation=78, fontsize=9, style="italic", color="#5d86a3",
        ha="center", va="center", zorder=2)
ax.text(mx(50.15), my(-19.2), "Indian Ocean", rotation=72, fontsize=9, style="italic", color="#5d86a3",
        ha="center", va="center", zorder=2)
ax.text(mx(46.55), my(-21.6), "MADAGASCAR", fontsize=9, color="#a39f94", ha="center", va="center",
        rotation=68, zorder=2, fontweight="bold")

# zoom-extent boxes
ext_of = {}
LAB_OFF = {"b": (0, 1, "center", "bottom"), "c": (1.5, 0, "left", "center"), "d": (0, -1, "center", "top"),
           "e": (1, 0, "left", "center"), "f": (1.5, 0, "left", "center")}
for letter, title, lonc, latc, span, (bx, by, bw, bh), km, step in ZOOMS:
    wm = mx(lonc + span / 2) - mx(lonc - span / 2); hm = wm * bh / bw
    x0 = mx(lonc) - wm / 2; yc = my(latc); ext_of[letter] = (x0, x0 + wm, yc - hm / 2, yc + hm / 2)
    ax.add_patch(Rectangle((x0, yc - hm / 2), wm, hm, fill=False, ec=INK, lw=0.8, zorder=8))
    ox, oy, ha, va = LAB_OFF[letter]
    px = x0 + wm / 2 + ox * wm / 2; py = yc + oy * hm / 2
    ax.annotate(letter, (px, py), xytext=(2.2 * ox, 2.2 * oy), textcoords="offset points", ha=ha, va=va,
                fontsize=10.5, fontweight="bold", color=INK, zorder=9, path_effects=halo)

# legend strip along the bottom of the figure
order = list(counts.index)
fig.text(X0 / W, (LEG - 3.2) / H, f"Sampling plots by project (number of plots; total n = {len(df):,})",
         fontsize=9, fontweight="bold", color=INK, ha="left", va="center")
handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=PROJ_COL[p], mec=EDGE, mew=0.5,
                  label=f"{nice(p)} ({counts[p]})") for p in order]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=((X0 - 1.5) / W, (LEG - 8.6) / H), ncol=6,
           fontsize=8.5, frameon=False, handletextpad=0.1, columnspacing=1.5, borderpad=0, borderaxespad=0)
fig.legend(handles=[Patch(fc=MANG, ec=MANG, label="Mangrove extent 2020 (Global Mangrove Watch v3.0)"),
                    Line2D([], [], color="#8fb8d4", lw=1.2, label="Major rivers"),
                    Line2D([], [], marker="s", ls="", ms=3.4, mfc="white", mec=INK, mew=0.6, label="Cities")],
           loc="center left", bbox_to_anchor=(X0 / W, (LEG - 14.0) / H), ncol=3, fontsize=8.5, frameon=False,
           handlelength=1.5, handletextpad=0.5, columnspacing=2.2, borderpad=0, borderaxespad=0)

# locator inset
iw = 27.0; ih = iw * 58 / 62
axi = add_ax(X0 + 1.8, Y0 + main_h - ih - 1.8, iw, ih); axi.set_facecolor("#f4f8fa")
world.plot(ax=axi, color="#d9d6cf", edgecolor="white", linewidth=0.15)
world[world.ADM0_A3 == "MDG"].plot(ax=axi, color="#3a3a3a", edgecolor="#3a3a3a", linewidth=0.2)
axi.add_patch(Rectangle((EXT[0], EXT[2]), EXT[1] - EXT[0], EXT[3] - EXT[2], fill=False, ec="#c0392b", lw=0.6))
axi.set_xlim(8, 70); axi.set_ylim(-38, 20); axi.set_aspect(1.0)
axi.set_xticks([]); axi.set_yticks([]); axi.set_xlabel(""); axi.set_ylabel("")
axi.text(24, 3, "AFRICA", fontsize=6.5, color="#7d7a73", ha="center", va="center")
axi.text(58, -30, "Indian\nOcean", fontsize=6, color="#6f8fa6", ha="center", va="center", style="italic",
         linespacing=0.95)
for s in axi.spines.values(): s.set_linewidth(0.5); s.set_edgecolor("#3a3a3a")

# ================= zoom panels =================
for letter, title, lonc, latc, span, (bx, by, bw, bh), km, step in ZOOMS:
    a = add_ax(bx, by, bw, bh)
    x0, x1, y0, y1 = ext_of[letter]
    bb = box(x0, y0, x1, y1).buffer((x1 - x0) * 0.05)
    land_c = land[land.intersects(bb)]; mang_c = mang[mang.intersects(bb)]
    land_c.plot(ax=a, color=LAND, edgecolor=COAST, linewidth=0.3, zorder=1)
    mang_c.plot(ax=a, color=MANG, edgecolor=MANG, linewidth=0.1, alpha=0.9, zorder=3)
    a.scatter(pts.geometry.x, pts.geometry.y, s=26, c=pts.Projet.map(PROJ_COL), edgecolors=EDGE,
              linewidths=0.45, zorder=6)
    a.set_xlim(x0, x1); a.set_ylim(y0, y1)
    inv = pyproj.Transformer.from_crs(CRS, 4326, always_xy=True)
    lo0, la0 = inv.transform(x0, y0); lo1, la1 = inv.transform(x1, y1)
    graticule(a, lo0, lo1, la0, la1, step, 0.5 if letter == "d" else None)
    scalebar(a, km, latc, loc="ll" if letter in ("b", "c") else "lr")
    panel_label(a, letter, title)
    n_in = int(((pts.geometry.x > x0) & (pts.geometry.x < x1) & (pts.geometry.y > y0) & (pts.geometry.y < y1)).sum())
    print(letter, title, "plots in panel:", n_in, "extent:", round(lo0, 3), round(lo1, 3), round(la0, 3), round(la1, 3))

area = gpd.read_file(GPKG).to_crs("ESRI:54034").area.sum() / 1e6
print("mangrove area km2:", round(area)); print(counts.to_string()); print("fig mm:", W, round(H, 1))
fig.savefig(f"{OUT}.pdf"); fig.savefig(f"{OUT}.png", dpi=600); print("map written to", OUT)
