# UTD19 Mini Notebook — Turn & Segment Flows (snippets)

> **Goal:** bite‑size, easy cells you can paste into a Jupyter notebook to explore the UTD‑19 dataset and get **hourly** segment counts and **turning-movement** estimates.
>
> **What you’ll get here**
>
> 1. Load & peek at `detectors.csv` / `links.csv`
> 2. Mark where detectors sit along a link (near stop line vs. just after intersection)
> 3. Chunk‑read the big UTD19 measurements, convert to **hourly vehicles**
> 4. Find an intersection that actually has detectors on both sides
> 5. Estimate **Left/Through/Right** flows for one node/hour with a tiny IPF (RAS)
> 6. Plot simple figures for **one node** and **one link**
>
> Keep it small first; scaling hints at the end.

---

## 0) (Optional) Install dependencies

```bash
pip install pandas numpy matplotlib geopandas shapely networkx
```

> You only *need* `pandas`, `numpy`, `matplotlib` for most of this. `geopandas/shapely/networkx` help with the tiny network bits but can be skipped if you’d rather just do segments.

---

## 1) Imports & paths

```python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "./data/UTD19"   # change me
CITY = "Zurich"              # change me (must match detectors/links citycode)
```

---

## 2) Load detectors & links (peek only)

```python
# --- detectors ---
det = pd.read_csv(os.path.join(DATA_DIR, "detectors.csv"))
det = det[det["citycode"] == CITY].copy()
print(det.shape)
det.head(3)
```

```python
# --- links ---
links = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
links = links[links["citycode"] == CITY].copy()
print(links.shape)
links.head(3)
```

> **Columns to notice**
>
> * detectors: `detid`, `linkid`, `lanes`, `length`, `pos`, `fclass` (plus citycode)
> * links: points along each link with `linkid`, `group`, `order`, `lat`, `long` (plus citycode)

---

## 2.1) Where is a detector on its link?

The manual defines a detector’s position along a link with `length` (km) and `pos` (km). We’ll use:

```python
# distance from the upstream end (km)
det["s_km"] = det["length"] - det["pos"]

# Flags for “near an intersection” (tune the thresholds if needed)
TOL_KM_NEAR_UPSTREAM   = 0.03  # ~30 m from upstream node (just after an intersection)
TOL_KM_NEAR_DOWNSTREAM = 0.03  # ~30 m from downstream node (approach stop line)

det["near_upstream"]   = det["s_km"] <= TOL_KM_NEAR_UPSTREAM
# equivalently: detectors with small pos are near downstream end
det["near_downstream"] = (det["length"] - det["s_km"]) <= TOL_KM_NEAR_DOWNSTREAM

# quick check
(det[["detid", "linkid", "s_km", "near_upstream", "near_downstream"]]
   .head(8))
```

---

## 3) Rebuild simple node IDs from links (no GIS required)

Links are given as polylines: one row per vertex with `group`/`order`. We’ll reconstruct a minimal start/end point per `(linkid, group)` and give each endpoint a stable string ID.

```python
# order vertices per link/group
links2 = links.sort_values(["linkid", "group", "order"])  

# pick first and last vertex per polyline segment
first = links2.groupby(["linkid", "group"]).first().reset_index()
last  = links2.groupby(["linkid", "group"]).last().reset_index()

# merge to get start/end
seg = first[["linkid", "group", "lat", "long"]].rename(columns={"lat":"lat0","long":"lon0"})
seg = seg.merge(last[["linkid", "group", "lat", "long"]].rename(columns={"lat":"lat1","long":"lon1"}),
                on=["linkid","group"], how="inner")

# helpers
def pt_id(lon, lat):
    return f"{lon:.6f},{lat:.6f}"

def bearing(p0, p1):
    lon1, lat1 = np.radians(p0)
    lon2, lat2 = np.radians(p1)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

seg["node_u"] = [pt_id(a,b) for a,b in zip(seg["lon0"], seg["lat0"])]
seg["node_v"] = [pt_id(a,b) for a,b in zip(seg["lon1"], seg["lat1"])]
seg["bearing_start"] = [bearing((a,b),(c,d)) for a,b,c,d in zip(seg["lon0"],seg["lat0"],seg["lon1"],seg["lat1"]) ]
seg["bearing_end"]   = seg["bearing_start"]  # good enough for first/last segment

seg.head(3)
```

> We’ll treat `node_u` as the **upstream** endpoint and `node_v` as the **downstream** endpoint in the driving direction stored in the file.

---

## 4) Join detectors to node roles

A detector on link L “belongs” to:

* `node_dn` (downstream node of L) if it’s **near\_downstream** → an **approach** detector.
* `node_up` (upstream node of L) if it’s **near\_upstream** → a **departure** detector.

```python
link_to_u = seg.set_index("linkid")["node_u"].to_dict()
link_to_v = seg.set_index("linkid")["node_v"].to_dict()

D = det[["detid","linkid","lanes","near_upstream","near_downstream","fclass"]].copy()
D["node_up"] = D["linkid"].map(link_to_u)
D["node_dn"] = D["linkid"].map(link_to_v)

# candidate nodes with at least one approach and one departure det
cand_nodes = (
    pd.concat([
        D.loc[D["near_downstream"], ["node_dn"]].rename(columns={"node_dn":"node"}),
        D.loc[D["near_upstream"],   ["node_up"]].rename(columns={"node_up":"node"})
    ], ignore_index=True)
      .value_counts().rename("n").reset_index()
)
print("candidate nodes:", len(cand_nodes))
cand_nodes.head(10)
```

---

## 5) Chunk‑read measurements → hourly vehicles for a tiny subset

We’ll take the **first 100 candidate detectors** found near a single node and compute **hourly** vehicles just for **one day** to keep it small.

```python
# pick one promising node
if len(cand_nodes):
    NODE = cand_nodes.loc[0, "node"]
else:
    NODE = None
NODE
```

```python
# small set of detectors near this node (both sides)
near_node = D.query("node_dn == @NODE and near_downstream or node_up == @NODE and near_upstream")
sel_detids = near_node["detid"].head(100).tolist()
len(sel_detids), sel_detids[:5]
```

```python
# choose a day that exists for your city (peek later if needed)
DAYS = None   # or like [20170605]

MEAS_PATH = os.path.join(DATA_DIR, "UTD19.csv")
if not os.path.exists(MEAS_PATH):
    MEAS_PATH = os.path.join(DATA_DIR, "utd19_u.csv")

# helper to infer interval length per (detid, day)
def infer_dt_seconds(df):
    diffs = np.diff(np.sort(df["interval"].unique()))
    diffs = diffs[diffs>0]
    if len(diffs)==0:
        return 300.0
    u = np.unique(diffs)[:3]
    return float(np.median(u))

# compute hourly vehicles for our small detector set
def hourly_counts_small(meas_path, city, detids, days=None, chunksize=2_000_000):
    rows = []
    for chunk in pd.read_csv(meas_path, chunksize=chunksize):
        c = chunk[(chunk["city"]==city) & (chunk["detid"].isin(detids))].copy()
        if days is not None:
            c = c[c["day"].isin(days)]
        if c.empty:
            continue
        c = c.dropna(subset=["flow"])  # flow is veh/h/lane
        dt_map = c.groupby(["detid","day"]).apply(infer_dt_seconds).rename("dt").reset_index()
        c = c.merge(dt_map, on=["detid","day"], how="left")
        lanes = D.set_index("detid")["lanes"].to_dict()
        c["lanes"] = c["detid"].map(lanes).fillna(1.0)
        c["veh"] = c["flow"].astype(float) * c["lanes"].astype(float) * (c["dt"].astype(float)/3600.0)
        c["hour"] = (c["interval"]//3600).astype(int)
        rows.append(c.groupby(["detid","day","hour"], as_index=False)["veh"].sum())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["detid","day","hour","veh"])

H = hourly_counts_small(MEAS_PATH, CITY, sel_detids, DAYS)
H.head(6)
```

---

## 6) Build approach (rows) and departure (cols) totals at our node

```python
# approach totals per incoming link (sum detectors near downstream)
inc = near_node[near_node["near_downstream"]].copy()
out = near_node[near_node["near_upstream"]].copy()

# hour to inspect (pick the most complete hour present)
keys = H.groupby(["day","hour"]).size().sort_values(ascending=False)
DAY, HOUR = (keys.index[0] if len(keys) else (None, None))
DAY, HOUR
```

```python
# row sums: incoming link → vehicles
row = (H.merge(inc[["detid","linkid"]], on="detid", how="inner")
        .query("day == @DAY and hour == @HOUR")
        .groupby("linkid", as_index=False)["veh"].sum())
row
```

```python
# col sums: outgoing link → vehicles
col = (H.merge(out[["detid","linkid"]], on="detid", how="inner")
        .query("day == @DAY and hour == @HOUR")
        .groupby("linkid", as_index=False)["veh"].sum())
col
```

---

## 7) Tiny IPF (RAS) to estimate a movement matrix

We’ll align incoming totals (rows) to outgoing totals (cols). No turn priors yet — just a neutral start.

```python
# create matrices
inc_links = row["linkid"].tolist()
out_links = col["linkid"].tolist()

R = row.set_index("linkid")["veh"].reindex(inc_links).fillna(0).to_numpy(float)
C = col.set_index("linkid")["veh"].reindex(out_links).fillna(0).to_numpy(float)

X = np.ones((len(R), len(C)), dtype=float)

# basic IPF
def ipf(X, R, C, iters=100, tol=1e-6):
    X = X.copy()
    for _ in range(iters):
        rs = X.sum(axis=1); rs[rs==0]=1
        X *= (R/rs)[:,None]
        cs = X.sum(axis=0); cs[cs==0]=1
        X *= (C/cs)[None,:]
        if (abs(X.sum(axis=1)-R).sum() + abs(X.sum(axis=0)-C).sum()) < tol*(R.sum()+C.sum()+1e-9):
            break
    return X

Xhat = ipf(X, R, C)

mov = (pd.DataFrame(Xhat, index=inc_links, columns=out_links)
         .rename_axis(index="in_link", columns="out_link")
         .stack().rename("veh").reset_index())
mov.head()
```

> **Interpretation:** each cell is the estimated vehicles from **incoming** link → **outgoing** link for the chosen hour.

---

## 8) Classify each movement as Left / Through / Right (L/T/R)

A quick angle-based classifier from the first/last segment bearings.

```python
# link bearings (very rough but serviceable)
B_in  = seg.set_index("linkid")["bearing_end"].to_dict()
B_out = seg.set_index("linkid")["bearing_start"].to_dict()

LEFT_MIN, LEFT_MAX   = 45.0, 135.0
RIGHT_MIN, RIGHT_MAX = -135.0, -45.0
U_TURN_MIN_ABS       = 150.0

def angle_diff(a_out, a_in):
    d = (a_out - a_in + 180) % 360 - 180
    return d

def turn_type(a_in, a_out):
    d = angle_diff(a_out, a_in)
    if abs(d) >= U_TURN_MIN_ABS: return "U"
    if LEFT_MIN <= d <= LEFT_MAX: return "L"
    if RIGHT_MIN <= d <= RIGHT_MAX: return "R"
    return "T"

mov["turn"] = [turn_type(B_in.get(i,0), B_out.get(o,0)) for i,o in zip(mov["in_link"], mov["out_link"])]
mov.sort_values("veh", ascending=False).head(10)
```

---

## 9) A tiny plot: L/T/R over the day for this node

Let’s repeat the IPF for every hour present (still just our tiny detector set) and plot.

```python
def node_turns_for_day(H, inc, out, day):
    res = []
    for hr in sorted(H[H["day"]==day]["hour"].unique()):
        row = (H.merge(inc[["detid","linkid"]], on="detid", how="inner")
                 .query("day == @day and hour == @hr").groupby("linkid")["veh"].sum())
        col = (H.merge(out[["detid","linkid"]], on="detid", how="inner")
                 .query("day == @day and hour == @hr").groupby("linkid")["veh"].sum())
        if row.empty and col.empty:
            continue
        R = row.reindex(inc["linkid"].unique()).fillna(0).to_numpy(float)
        C = col.reindex(out["linkid"].unique()).fillna(0).to_numpy(float)
        if R.sum()==0 and C.sum()>0: R = np.full_like(R, C.sum()/max(len(R),1))
        if C.sum()==0 and R.sum()>0: C = np.full_like(C, R.sum()/max(len(C),1))
        X = ipf(np.ones((len(R),len(C))), R, C)
        tmp = (pd.DataFrame(X, index=inc["linkid"].unique(), columns=out["linkid"].unique())
                 .rename_axis(index="in_link", columns="out_link").stack().rename("veh").reset_index())
        tmp["turn"] = [turn_type(B_in.get(i,0), B_out.get(o,0)) for i,o in zip(tmp["in_link"], tmp["out_link"])]
        tmp = tmp.groupby("turn")["veh"].sum().reindex(["L","T","R","U"]).fillna(0)
        tmp = tmp.to_frame().T.assign(hour=hr)
        res.append(tmp)
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame()

TS = node_turns_for_day(H, inc, out, DAY)
TS.head()
```

```python
# Plot (matplotlib default styling, one line per turn)
plt.figure(figsize=(8,4))
for k in ["L","T","R","U"]:
    if k in TS.columns:
        plt.plot(TS["hour"], TS[k], label=k, linewidth=2)
plt.title(f"Node {NODE}\nEstimated turning flows by hour (day {DAY})")
plt.xlabel("Hour")
plt.ylabel("Vehicles / hour")
plt.legend(title="Turn")
plt.tight_layout()
plt.show()
```

---

## 10) Segment (link) hourly counts — the simplest view

If turns feel too magical, start with plain **segment** counts (sum of detectors on the link).

```python
seg_hour = (H.merge(D[["detid","linkid"]], on="detid", how="left")
              .groupby(["linkid","day","hour"], as_index=False)["veh"].sum())

one_link = seg_hour.sort_values("veh", ascending=False)["linkid"].iloc[0]
seg_one  = seg_hour[seg_hour["linkid"]==one_link]
seg_one.head()
```

```python
plt.figure(figsize=(8,4))
for day, g in seg_one.groupby("day"):
    plt.plot(g["hour"], g["veh"], label=str(day), linewidth=2)
plt.title(f"Link {one_link}: hourly vehicles by day")
plt.xlabel("Hour")
plt.ylabel("Vehicles / hour")
plt.legend(title="Day", ncols=2)
plt.tight_layout()
plt.show()
```

---

## Scaling up (when you’re ready)

* Increase the set of `sel_detids` (e.g., *all* detids within a radius of a corridor) or drop the limit entirely.
* Loop over hours/days and write tidy CSVs (node, in\_link, out\_link, hour, veh, turn).
* Add **turn priors**: initialize `X` with larger weights for **Through** on arterials, etc., or disallow **U‑turns** by zeroing those cells.
* Bucket by road class using the detectors’ `fclass` (map to **highway/arterial/residential** buckets if you want class‑pair stats like highway→arterial).
* If you install `geopandas`, you can map top intersections with two lines.

> Want me to turn this into a real `.ipynb` with outputs saved to disk? I can package these cells and add a few helper widgets next.

---

## ✨ Optional: install prettier plotting packages

```bash
pip install seaborn contextily geopandas shapely pyproj
```

> `seaborn` = nicer defaults; `contextily` = basemap tiles; `geopandas/shapely` = quick geometries; `pyproj` = reprojection.

---

## A) Set up Seaborn theme (one-liner)

```python
import seaborn as sns
sns.set_theme(context="notebook", style="whitegrid")  # try context="talk" for slide-ready
```

---

## B) Prettier L/T/R time series with Seaborn

*(drop this cell right after you compute `TS` in section 9)*

```python
# TS has columns like ['hour','L','T','R','U'] (some may be missing)
cols = [c for c in ["L","T","R","U"] if c in TS.columns]
plot_df = TS.melt(id_vars="hour", value_vars=cols, var_name="turn", value_name="veh")

ax = sns.lineplot(data=plot_df, x="hour", y="veh", hue="turn", marker="o")
ax.set_title(f"Node {NODE}
Estimated turning flows by hour (day {DAY})")
ax.set_xlabel("Hour")
ax.set_ylabel("Vehicles / hour")
ax.figure.set_size_inches(8, 4)
plt.tight_layout()
plt.show()
```

---

## C) L/T/R heatmap (compact overview)

```python
heat = (TS.set_index("hour")[cols].fillna(0)).T  # rows=turn, cols=hour
ax = sns.heatmap(heat, annot=False, cbar_kws={"label":"veh/h"})
ax.set_title(f"Node {NODE}: L/T/R intensity by hour (day {DAY})")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Turn")
ax.figure.set_size_inches(7, 3.6)
plt.tight_layout()
plt.show()
```

---

## D) Nicer segment plot for one link

*(drop this cell after section 10 where `seg_one` is built)*

```python
ax = sns.lineplot(data=seg_one, x="hour", y="veh", hue="day", marker="o")
ax.set_title(f"Link {one_link}: hourly vehicles by day")
ax.set_xlabel("Hour")
ax.set_ylabel("Vehicles / hour")
ax.figure.set_size_inches(8, 4)
ax.legend(title="Day", ncols=2, frameon=True)
plt.tight_layout()
plt.show()
```

---

## E) Tiny basemap around the selected node (Contextily)

> Plots incoming vs outgoing links for `NODE` over a light basemap. Requires internet access for tiles.

```python
import geopandas as gpd
from shapely.geometry import LineString, Point
import contextily as cx

# Build mini GeoDataFrames for incoming/outgoing links touching NODE
inc_ids = inc["linkid"].unique() if len(inc) else []
out_ids = out["linkid"].unique() if len(out) else []

sub_inc = seg[seg["linkid"].isin(inc_ids)].copy()
sub_out = seg[seg["linkid"].isin(out_ids)].copy()

# Make simple straight lines using first/last vertices
make_lines = lambda g: gpd.GeoDataFrame(
    g[["linkid"]].copy(),
    geometry=[LineString([(a,b),(c,d)]) for a,b,c,d in zip(g["lon0"], g["lat0"], g["lon1"], g["lat1"])],
    crs="EPSG:4326"
)
Ginc = make_lines(sub_inc).to_crs(3857)
Gout = make_lines(sub_out).to_crs(3857)
Gpt  = gpd.GeoDataFrame({"node":[NODE]},
                        geometry=[Point(float(NODE.split(",")[0]), float(NODE.split(",")[1]))],
                        crs="EPSG:4326").to_crs(3857)

# Plot
fig, ax = plt.subplots(figsize=(7,7))
if len(Ginc): Ginc.plot(ax=ax, linewidth=3, alpha=0.85, color="#d81b60", label="incoming")
if len(Gout): Gout.plot(ax=ax, linewidth=3, alpha=0.85, color="#1e88e5", label="outgoing")
Gpt.plot(ax=ax, color="#2e7d32", markersize=60, zorder=5, label="node")

cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
ax.set_title(f"Node {NODE}: incoming (pink) vs outgoing (blue)")
ax.axis("off")
ax.legend(loc="lower left")
plt.tight_layout()
plt.show()
```

---

## F) Save figures with nice margins

```python
# After any plt.show(), you can also save:
plt.savefig("out/figures/example_plot.png", dpi=200, bbox_inches="tight")
```

---

### Tips

* Try `sns.set_theme(context="talk", style="whitegrid")` for presentation-sized fonts.
* If basemap tiles don’t load (firewall/no internet), omit `cx.add_basemap(ax)` — your geometries still draw.
* Want categorical palettes? Add `palette="tab10"` to `sns.lineplot(...)`.
