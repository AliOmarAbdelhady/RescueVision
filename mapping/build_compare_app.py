"""
Build the compare view (before|after side by side, REAL vs GENERATED map
layers, synced pan/zoom) and the original dataset browser page.

Outputs (all in output/, served by start_map_server.sh):
  compare.html   two synced maps; each can show the raw imagery ("REAL") or
                 imagery + the change-detection overlay ("GENERATED")
  dataset.html   the ORIGINAL Maxar scenes covering this area (thumbnails,
                 dates, cloud %, links to the full-resolution files)
"""

import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).parent
OUT = ROOT / "output"
THUMBS = OUT / "thumbs"
THUMBS.mkdir(exist_ok=True)

LAT_MIN, LAT_MAX = 37.5724036, 37.5795980
LON_MIN, LON_MAX = 36.9174790, 36.9265152
BOUNDS = [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]]

S3 = "https://maxar-opendata.s3.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23"

# ------------------------------------------------------- original dataset info
rows = pq.read_table(ROOT / "data" / "turkey-earthquake.parquet").to_pylist()
lon, lat = 36.922, 37.576          # city-center test point
scenes = [r for r in rows
          if r["bbox"][0] <= lon <= r["bbox"][2] and r["bbox"][1] <= lat <= r["bbox"][3]]
scenes.sort(key=lambda r: r["datetime"])
print(f"event total tiles: {len(rows)} | covering city center: {len(scenes)}")

# ------------------------------------------------------------- thumbnails
def thumb_url(r):
    cat = r["id"].split("/")[-1]
    date = r["datetime"].date()
    return f"{S3}/ard/{r['utm_zone']}/{r['quadkey']}/{date}/{cat}-visual.tif"

for r in scenes:
    cat = r["id"].split("/")[-1]
    date = str(r["datetime"].date())
    dest = THUMBS / f"{date}_{cat}.png"
    if dest.exists():
        continue
    subprocess.run(
        ["gdal_translate", "-q", "-of", "PNG", "-outsize", "512", "0",
         "/vsicurl/" + thumb_url(r), str(dest)], check=True)
    print("thumb:", dest.name)

# ---------------------------------------------------------------- compare.html
compare = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compare — Kahramanmaras before/after</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body {{ margin:0; height:100%; background:#141414; font-family:sans-serif; }}
  #wrap {{ display:flex; height:calc(100% - 92px); }}
  .pane {{ width:50%; display:flex; flex-direction:column; }}
  .pane .map {{ flex:1; }}
  .bar {{ display:flex; align-items:center; gap:8px; padding:6px 10px; background:#1d1d1d; color:#ddd; }}
  .bar b {{ font-size:13px; }}
  .seg {{ display:flex; overflow:hidden; border-radius:6px; border:1px solid #444; }}
  .seg button {{ background:#2a2a2a; color:#bbb; border:0; padding:5px 10px; font-size:12px; cursor:pointer; }}
  .seg button.on {{ background:#0a84ff; color:#fff; }}
  #top {{ background:#111; color:#eee; padding:8px 12px; font-size:13px; }}
  #top a {{ color:#7cc0ff; margin-right:14px; text-decoration:none; }}
  #foot {{ background:#111; color:#999; padding:6px 12px; font-size:11px; }}
  .leaflet-container {{ background:#141414; }}
  @media (max-width:820px) {{ #wrap {{ flex-direction:column; }} .pane {{ width:100%; }} }}
</style>
</head>
<body>
<div id="top">
  <a href="satellite_city_map.html">single map</a><a href="compare.html"><b>compare</b></a>
  <a href="process.html">process</a>
  <a href="dataset.html">original dataset</a>
  <span style="float:right;color:#888">imagery: Maxar Open Data · 0.5 m/px</span>
</div>
<div id="wrap">
  <div class="pane" id="paneL">
    <div class="bar"><b>BEFORE — 26 Jul 2022</b>
      <span class="seg" data-pane="L">
        <button data-mode="real" class="on">REAL</button>
        <button data-mode="gen">+ GENERATED</button>
      </span>
    </div>
    <div class="map" id="mapL"></div>
  </div>
  <div class="pane" id="paneR">
    <div class="bar"><b>AFTER — 28 Feb 2023 (22 days post-earthquake)</b>
      <span class="seg" data-pane="R">
        <button data-mode="real" class="on">REAL</button>
        <button data-mode="gen">+ GENERATED</button>
      </span>
    </div>
    <div class="map" id="mapR"></div>
  </div>
</div>
<div id="foot">GENERATED layer = real pixel-change detection between the two dates (baseline method, not trained AI —
the xBD-trained model will replace it). Red = changed clusters. &nbsp;🟥 changed area</div>
<script>
const BOUNDS = {json.dumps(BOUNDS)};
function mkMap(id, image) {{
  const m = L.map(id, {{ minZoom: 15, maxZoom: 19, maxBounds: BOUNDS, maxBoundsViscosity: 1.0 }});
  m.fitBounds(BOUNDS);
  L.imageOverlay(image, BOUNDS).addTo(m);
  L.control.scale({{ imperial: false }}).addTo(m);
  return m;
}}
const mapL = mkMap('mapL', 'city_before.png');
const mapR = mkMap('mapR', 'city_after.png');
const dmgL = L.imageOverlay('damage_overlay.png', BOUNDS);
const dmgR = L.imageOverlay('damage_overlay.png', BOUNDS);

// keep the two panes in lock-step
let syncing = false;
function bindSync(a, b) {{
  a.on('move', () => {{
    if (syncing) return; syncing = true;
    b.setView(a.getCenter(), a.getZoom(), {{ animate: false }});
    syncing = false;
  }});
}}
bindSync(mapL, mapR); bindSync(mapR, mapL);

// REAL / +GENERATED toggles
document.querySelectorAll('.seg').forEach(seg => {{
  const pane = seg.dataset.pane;
  seg.querySelectorAll('button').forEach(btn => {{
    btn.onclick = () => {{
      seg.querySelectorAll('button').forEach(x => x.classList.remove('on'));
      btn.classList.add('on');
      const dmg = pane === 'L' ? dmgL : dmgR;
      const map = pane === 'L' ? mapL : mapR;
      if (btn.dataset.mode === 'gen') {{ if (!map.hasLayer(dmg)) dmg.addTo(map); }}
      else map.removeLayer(dmg);
    }};
  }});
}});
mapL.setView([(BOUNDS[0][0]+BOUNDS[1][0])/2, (BOUNDS[0][1]+BOUNDS[1][1])/2], 16, {{animate:false}});
</script>
</body>
</html>"""
(OUT / "compare.html").write_text(compare)
print("wrote compare.html")

# ---------------------------------------------------------------- dataset.html
cards = []
for r in scenes:
    cat = r["id"].split("/")[-1]
    date = str(r["datetime"].date())
    tag = "PRE-disaster" if date < "2023-02-06" else "POST-disaster"
    color = "#0a84ff" if tag == "PRE-disaster" else "#e3342f"
    cards.append(f"""
  <div class="card">
    <a href="{S3}/ard/{r['utm_zone']}/{r['quadkey']}/{date}/{cat}-visual.tif" target="_blank">
      <img src="thumbs/{date}_{cat}.png" alt="{cat}"></a>
    <div class="meta">
      <span class="tag" style="background:{color}">{tag}</span> <b>{date}</b><br>
      catalog: <code>{cat}</code><br>
      clouds: {r['tile:clouds_percent']}% · tile: {r['utm_zone']}/{r['quadkey']}<br>
      <a href="{S3}/ard/{r['utm_zone']}/{r['quadkey']}/{date}/{cat}-visual.tif" target="_blank">open full GeoTIFF ↗</a>
    </div>
  </div>""")

dataset = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Original dataset — Maxar Open Data, Turkiye earthquake 23</title>
<style>
  html,body {{ margin:0; background:#141414; color:#ddd; font-family:sans-serif; }}
  #top {{ background:#111; padding:8px 12px; font-size:13px; }}
  #top a {{ color:#7cc0ff; margin-right:14px; text-decoration:none; }}
  header {{ padding:18px 22px 4px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  p.sub {{ color:#999; font-size:13px; margin:0 0 14px; }}
  #grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:16px; padding:10px 22px 30px; }}
  .card {{ background:#1d1d1d; border-radius:10px; overflow:hidden; }}
  .card img {{ width:100%; display:block; }}
  .meta {{ padding:10px 12px; font-size:12px; line-height:1.6; color:#bbb; }}
  .tag {{ color:#fff; font-size:11px; padding:2px 7px; border-radius:4px; }}
  code {{ color:#8fd3ff; }}
  .note {{ padding:0 22px 16px; color:#888; font-size:12px; }}
  a {{ color:#7cc0ff; }}
</style>
</head>
<body>
<div id="top">
  <a href="satellite_city_map.html">single map</a><a href="compare.html">compare</a>
  <a href="process.html">process</a>
  <a href="dataset.html"><b>original dataset</b></a>
</div>
<header>
  <h1>Original dataset — every satellite scene covering this city area</h1>
  <p class="sub">Event: Kahramanmaras Turkiye earthquake 23 · Maxar Open Data (free, no registration) ·
  this event has {len(rows)} image tiles in total; {len(scenes)} of them cover our city-center area, spanning pre- and post-disaster dates.</p>
</header>
<div class="note">Thumbnails are real overviews of the original GeoTIFFs. Click any card to open the full-resolution
source file on the Maxar Open Data bucket. <a href="{S3}/" target="_blank">Browse the whole event bucket ↗</a> ·
<a href="https://www.maxar.com/open-data" target="_blank">about Maxar Open Data ↗</a></div>
<div id="grid">{''.join(cards)}</div>
</body>
</html>"""
(OUT / "dataset.html").write_text(dataset)
print("wrote dataset.html")
