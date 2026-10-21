"""
Build the PROCESS view: collect individual satellite images -> assemble them
into one 2D map -> final zoomed map with BEFORE / AFTER / +GENERATED views.

Uses only real data: the six Feb-28-2023 Maxar tiles (main post-disaster pass)
that cover the Kahramanmaras city region, plus our high-res city-center crops.

Output: output/process.html   (nav entry added to the other pages too)
"""

import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).parent
OUT = ROOT / "output"
THUMBS = OUT / "thumbs"
THUMBS.mkdir(exist_ok=True)

S3 = "https://maxar-opendata.s3.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23"
CATALOG = "10300100E3154100"          # main 28 Feb 2023 post-disaster pass
DATE = "2023-02-28"
CITY = [37.5724036, 36.9174790, 37.5795980, 36.9265152]   # lat0, lon0, lat1, lon1

rows = pq.read_table(ROOT / "data" / "turkey-earthquake.parquet").to_pylist()
sel = []
for r in rows:
    x0, y0, x1, y1 = r["bbox"]
    if (str(r["datetime"].date()) == DATE and r["id"].split("/")[-1] == CATALOG
            and r["tile:clouds_percent"] is not None and r["tile:clouds_percent"] <= 10
            and not (x1 < 36.88 or x0 > 37.04 or y1 < 37.50 or y0 > 37.65)):
        sel.append(r)
sel.sort(key=lambda r: (-r["bbox"][1], r["bbox"][0]))     # north row first, west first
print(f"mosaic tiles: {len(sel)}")

# thumbnails (real overviews of the original GeoTIFFs)
scenes = []
for r in sel:
    x0, y0, x1, y1 = r["bbox"]
    quad = r["quadkey"]
    dest = THUMBS / f"{DATE}_{quad}_{CATALOG}.png"
    if not dest.exists():
        subprocess.run(["gdal_translate", "-q", "-of", "PNG", "-outsize", "512", "0",
                        "/vsicurl/" + f"{S3}/ard/{r['utm_zone']}/{quad}/{DATE}/{CATALOG}-visual.tif",
                        str(dest)], check=True)
    scenes.append({
        "quad": quad,
        "bounds": [[y0, x0], [y1, x1]],
        "center": [(y0 + y1) / 2, (x0 + x1) / 2],
        "thumb": f"thumbs/{dest.name}",
        "clouds": r["tile:clouds_percent"],
    })

# cumulative coverage % (area union of placed tiles / final union of all tiles)
def union_area(rects):
    xs = sorted({b[0] for b in rects} | {b[2] for b in rects})
    ys = sorted({b[1] for b in rects} | {b[3] for b in rects})
    a = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(b[0] - 1e-9 <= cx <= b[2] + 1e-9 and b[1] - 1e-9 <= cy <= b[3] + 1e-9
                   for b in rects):
                a += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return a

all_rects = [r["bbox"] for r in sel]
total = union_area(all_rects)
ux0 = min(b[0] for b in all_rects); uy0 = min(b[1] for b in all_rects)
ux1 = max(b[2] for b in all_rects); uy1 = max(b[3] for b in all_rects)
union = [ux0, uy0, ux1, uy1]
placed = []
cum = []
for r in sel:
    placed.append(r["bbox"])
    cum.append(round(100 * union_area(placed) / total))

union_bounds = [[union[1], union[0]], [union[3], union[2]]]

# ---------------------------------------------------------------- process.html
SCENES_JS = json.dumps(scenes, ensure_ascii=False)
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Process — from collected images to a 2D map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%;background:#141414;color:#ddd;font-family:sans-serif}
  #top{background:#111;padding:8px 12px;font-size:13px}
  #top a{color:#7cc0ff;margin-right:14px;text-decoration:none}
  #steps{display:flex;gap:10px;align-items:center;background:#1b1b1b;padding:8px 12px;
         border-bottom:1px solid #2c2c2c;flex-wrap:wrap}
  .step{padding:5px 12px;border-radius:6px;background:#2a2a2a;color:#888;font-size:12px}
  .step.active{background:#0a84ff;color:#fff}
  .step.done{background:#1f7a34;color:#dff5e3}
  #play{background:#e39b26;border:0;color:#221a05;font-weight:700;padding:7px 18px;
        border-radius:6px;font-size:13px;cursor:pointer}
  #cov{margin-left:auto;font-size:12px;color:#9ad}
  #main{display:flex;height:calc(100% - 96px)}
  #map{flex:1}
  #cards{width:190px;background:#181818;overflow-y:auto;padding:8px}
  .card{background:#222;border-radius:8px;margin-bottom:8px;padding:6px;font-size:11px;color:#aaa}
  .card img{width:100%;border-radius:4px;display:block}
  .card .st{float:right}
  .card.placed{outline:2px solid #1f7a34}
  .card.placed .st{color:#6fe08a}
  #final{position:absolute;right:14px;top:104px;z-index:1000;display:none;gap:6px}
  #final button{background:#2a2a2a;color:#bbb;border:1px solid #444;border-radius:6px;
                padding:6px 12px;font-size:12px;cursor:pointer}
  #final button.on{background:#0a84ff;color:#fff;border-color:#0a84ff}
  #note{position:absolute;left:14px;bottom:26px;z-index:1000;background:#111d;color:#bbb;
        padding:6px 10px;border-radius:6px;font-size:11px}
  .leaflet-container{background:#101010}
  @media (max-width:760px){#main{flex-direction:column}#cards{width:auto;display:flex;
        overflow-x:auto;gap:8px}.card{min-width:130px;margin:0}}
</style>
</head>
<body>
<div id="top">
  <a href="satellite_city_map.html">single map</a><a href="compare.html">compare</a>
  <a href="dataset.html">original dataset</a><a href="process.html"><b>process</b></a>
</div>
<div id="steps">
  <span class="step" id="s1">1 COLLECT 0/__N__</span>
  <span class="step" id="s2">2 ASSEMBLE</span>
  <span class="step" id="s3">3 FINAL MAP</span>
  <button id="play">▶ PLAY</button>
  <span id="cov">coverage 0%</span>
</div>
<div id="main">
  <div style="position:relative;flex:1;display:flex">
    <div id="map"></div>
    <div id="final">
      <button id="bBefore">BEFORE</button>
      <button id="bAfter" class="on">AFTER</button>
      <button id="bDmg">+ GENERATED</button>
      <button id="bReplay">↺ replay</button>
    </div>
    <div id="note">images: Maxar Open Data · 28 Feb 2023 pass · 0.5 m/px</div>
  </div>
  <div id="cards"></div>
</div>
<script>
const SCENES = __SCENES__;
const CITY = __CITYBOUNDS__;
const UNION = __UNION__;
const CUM = __CUM__;

const map = L.map('map', {minZoom: 10, maxZoom: 19});
map.fitBounds(UNION);
L.control.scale({imperial:false}).addTo(map);

const cards = document.getElementById('cards');
SCENES.forEach((s, i) => {
  const d = document.createElement('div');
  d.className = 'card'; d.id = 'card' + i;
  d.innerHTML = '<span class="st">waiting</span><b>image ' + (i+1) + '</b><br>' +
    'tile ' + s.quad.slice(-3) + ' · clouds ' + s.clouds + '%<br>' +
    '<img src="' + s.thumb + '">';
  cards.appendChild(d);
});

let placed = [];
function place(i) {
  const s = SCENES[i];
  const ov = L.imageOverlay(s.thumb, s.bounds, {opacity: 0}).addTo(map);
  let o = 0;
  const t = setInterval(() => { o += 0.08; if (o >= 1) {o = 1; clearInterval(t);} ov.setOpacity(o); }, 30);
  placed.push(s);
  const c = document.getElementById('card' + i);
  c.classList.add('placed'); c.querySelector('.st').textContent = 'placed ✓';
  document.getElementById('s1').textContent = '1 COLLECT ' + (i+1) + '/' + SCENES.length;
  document.getElementById('s1').classList.add('active');
  document.getElementById('cov').textContent = 'coverage ' + CUM[i] + '%';
}

let playing = false;
async function play() {
  if (playing) return; playing = true;
  document.getElementById('play').disabled = true;
  for (let i = 0; i < SCENES.length; i++) {
    const s = SCENES[i];
    map.flyToBounds(s.bounds, {duration: 1.1, padding: [30, 30]});
    await new Promise(r => setTimeout(r, 1250));
    place(i);
    await new Promise(r => setTimeout(r, 550));
  }
  document.getElementById('s1').classList.remove('active');
  document.getElementById('s1').classList.add('done');
  document.getElementById('s2').classList.remove('active');
  document.getElementById('s2').classList.add('done');
  // zoom to the city and fade in the high-res final map
  document.getElementById('s3').classList.add('active');
  map.flyToBounds(CITY, {duration: 1.6, padding: [20, 20]});
  await new Promise(r => setTimeout(r, 1800));
  hi = L.imageOverlay('city_after.png', CITY, {opacity: 0}).addTo(map);
  let o = 0;
  const t = setInterval(() => { o += 0.05; if (o >= 1) {o = 1; clearInterval(t);} hi.setOpacity(o); }, 40);
  await new Promise(r => setTimeout(r, 900));
  document.getElementById('s3').classList.remove('active');
  document.getElementById('s3').classList.add('done');
  document.getElementById('final').style.display = 'flex';
}

let hi = null, hiDmg = null;
const bB = document.getElementById('bBefore'), bA = document.getElementById('bAfter'),
      bD = document.getElementById('bDmg');
function setHi(which) {
  if (hi) map.removeLayer(hi);
  hi = L.imageOverlay(which === 'before' ? 'city_before.png' : 'city_after.png', CITY).addTo(map);
  bB.classList.toggle('on', which === 'before');
  bA.classList.toggle('on', which !== 'before');
}
bB.onclick = () => setHi('before');
bA.onclick = () => setHi('after');
bD.onclick = () => {
  if (hiDmg && map.hasLayer(hiDmg)) { map.removeLayer(hiDmg); bD.classList.remove('on'); }
  else { hiDmg = L.imageOverlay('damage_overlay.png', CITY).addTo(map); bD.classList.add('on'); }
};
document.getElementById('bReplay').onclick = () => location.reload();
document.getElementById('play').onclick = play;
if (location.search.includes('autostart')) setTimeout(play, 400);
</script>
</body>
</html>"""

html = (HTML
        .replace("__SCENES__", SCENES_JS)
        .replace("__CITYBOUNDS__", json.dumps([[CITY[0], CITY[1]], [CITY[2], CITY[3]]]))
        .replace("__UNION__", json.dumps(union_bounds))
        .replace("__CUM__", json.dumps(cum))
        .replace("__N__", str(len(scenes))))
(OUT / "process.html").write_text(html)
print("wrote process.html | coverage steps:", cum)
