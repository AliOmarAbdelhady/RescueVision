#!/usr/bin/env bash
# Serve the SAR map on the local network (view from phone: http://<this-ip>:8000/satellite_city_map.html)
cd "$(dirname "$0")/output" && python3 -m http.server 8000
