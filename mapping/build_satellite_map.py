"""
Build the themed 2D satellite map (Google-Maps-style viewer) for the SAR grad project.

Inputs : output/city_before.png, output/city_after.png
         Real Maxar satellite images of Kahramanmaras city center (Turkiye),
         cropped from the open-data earthquake set, GPS-referenced.
Output : output/satellite_city_map.html  (open in any browser, incl. phone)

Theme: before/after toggle, base-map switcher (Esri imagery / OSM), scale bar,
legend with the damage classes the AI module will fill in later, and one
clearly-marked example annotation showing where AI results will go.
"""

from pathlib import Path

import folium

OUT = Path(__file__).parent / "output"

LAT_MIN, LAT_MAX = 37.5724036, 37.5795980
LON_MIN, LON_MAX = 36.9174790, 36.9265152
CENTER = [(LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2]

m = folium.Map(
    location=CENTER, zoom_start=17, min_zoom=16, tiles=None, control_scale=True,
    max_bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],   # panning stays inside our image
    no_wrap=True,
)
m.get_root().html.add_child(folium.Element(
    "<style>.leaflet-container{background:#141414}</style>"))  # dark frame, not gray "infinite paper"

# (No world basemaps — the map is ONLY our imagery.)

folium.raster_layers.ImageOverlay(
    "city_before.png",
    bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
    opacity=1.0, name="BEFORE — 26 Jul 2022 (Maxar)", show=False, z_index=50,
).add_to(m)
folium.raster_layers.ImageOverlay(
    "city_after.png",
    bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
    opacity=1.0, name="AFTER — 28 Feb 2023, 22 days post-earthquake (Maxar)", z_index=100,
).add_to(m)

folium.LayerControl(collapsed=False, position="topright").add_to(m)

# Make the two image layers mutually exclusive: clicking BEFORE shows that
# full image and hides AFTER (and vice versa) — a clean click-to-switch view.
m.get_root().html.add_child(folium.Element("""
<script>
(function(){
  function once(fn){ if(window.map) fn(); else setTimeout(()=>once(fn), 200); }
  once(function(){
    const layers = Object.values(map._layers).filter(l => l._url && l._url.includes('city_'));
    const [before, after] = layers;
    map.on('overlayadd', function(e){
      if(e.layer===before) map.removeLayer(after);
      if(e.layer===after)  map.removeLayer(before);
    });
  });
})();
</script>"""))

# Full-view buttons: open each raw image full-size in its own tab.
m.get_root().html.add_child(folium.Element("""
<div style="position:fixed;bottom:22px;right:12px;z-index:9999;display:flex;gap:8px">
  <a href="city_before.png" target="_blank"
     style="background:#111d;color:#eee;padding:8px 12px;border-radius:8px;
            font-family:sans-serif;font-size:13px;text-decoration:none;border:1px solid #444">
     &#x26F6; full view — BEFORE</a>
  <a href="city_after.png" target="_blank"
     style="background:#111d;color:#eee;padding:8px 12px;border-radius:8px;
            font-family:sans-serif;font-size:13px;text-decoration:none;border:1px solid #444">
     &#x26F6; full view — AFTER</a>
</div>"""))


legend = """
<div style="position:fixed;bottom:22px;left:12px;z-index:9999;background:#ffffffee;
     border-radius:8px;padding:10px 14px;font-family:sans-serif;font-size:13px;
     box-shadow:0 1px 6px rgba(0,0,0,.35);line-height:1.55">
  <b>Legend</b><br>
  <span style="display:inline-block;width:12px;height:12px;background:#e3342f;
       opacity:.6;border:1px solid #e3342f;border-radius:2px"></span> destroyed&nbsp;
  <span style="display:inline-block;width:12px;height:12px;background:#f39c12;
       opacity:.6;border:1px solid #f39c12;border-radius:2px"></span> major&nbsp;
  <span style="display:inline-block;width:12px;height:12px;background:#f7e04b;
       opacity:.6;border:1px solid #c9b437;border-radius:2px"></span> minor<br>
  <span style="display:inline-block;width:12px;height:12px;background:#00bcd4;
       border-radius:50%"></span> entry route&nbsp;
  <span style="display:inline-block;width:12px;height:12px;background:#00c853;
       border-radius:50%"></span> exit route<br>
  <span style="color:#888">damage classes &amp; routes fill in as AI modules run</span>
</div>"""
m.get_root().html.add_child(folium.Element(legend))

title = """
<div style="position:fixed;top:10px;left:50px;z-index:9999;background:#111d;
     color:#eee;padding:9px 16px;border-radius:8px;font-family:sans-serif;font-size:14px">
  <b>Kahramanmaras city center — Turkiye earthquake 23 · SAR grad project</b><br>
  <span style="font-size:12px;color:#bbb">imagery: Maxar Open Data · 0.5 m/px · ~0.8 x 0.8 km ·
  toggle BEFORE / AFTER in the layers panel<br>
  views: <a href="satellite_city_map.html" style="color:#7cc0ff">single map</a> ·
  <a href="compare.html" style="color:#7cc0ff">side-by-side compare</a> ·
  <a href="process.html" style="color:#7cc0ff">process</a> ·
  <a href="dataset.html" style="color:#7cc0ff">original dataset</a></span>
</div>"""
m.get_root().html.add_child(folium.Element(title))

m.save(OUT / "satellite_city_map.html")

# Folium embeds local images as base64 inside the HTML (~31 MB). Swap them for
# relative file references so the browser fetches each PNG only when its layer
# is shown (HTML drops to ~30 KB). Occurrence order = layer add order.
html = (OUT / "satellite_city_map.html").read_text()
parts = html.split("data:image/png;base64,")
assert len(parts) == 3, f"expected 2 embedded images, found {len(parts) - 1}"
names = ["city_before.png", "city_after.png"]
fixed = parts[0]
for name, part in zip(names, parts[1:]):
    _payload, quote, rest = part.partition('"')   # drop the base64 blob
    fixed += name + quote + rest
(OUT / "satellite_city_map.html").write_text(fixed)
print("saved:", OUT / "satellite_city_map.html")
