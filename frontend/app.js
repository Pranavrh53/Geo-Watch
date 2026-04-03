// Satellite Change Detection — Frontend Application
// ─────────────────────────────────────────────────
// Requires: Leaflet.js already loaded in index.html
//           A logged-in JWT token stored in localStorage as "auth_token"

const API_BASE = 'http://localhost:8000';

// ── Auth helper ──────────────────────────────────────────────────────────────
function authHeaders() {
  const token = localStorage.getItem('auth_token');
  return {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
  };
}

// ── Legend config (single source of truth) ───────────────────────────────────
const LEGEND = [
  { key: 'construction',       label: 'New Construction',     emoji: '🏗️',  color: '#FF8C00' },
  { key: 'deforestation',      label: 'Deforestation',        emoji: '🌲',  color: '#DC1E1E' },
  { key: 'vegetation_growth',  label: 'Vegetation Growth',    emoji: '🌱',  color: '#28B446' },
  { key: 'water_loss',         label: 'Water Body Shrinking', emoji: '💧',  color: '#2878FF' },
  { key: 'roads',              label: 'New Roads',            emoji: '🛣️',  color: '#444444' },
];

// ── Global state ─────────────────────────────────────────────────────────────
let map;
let currentCity      = null;
let citiesData       = null;
let overlayLayer     = null;   // current L.imageOverlay on map
let cityRect         = null;   // bounding-box rectangle
let lastResults      = null;   // last API response

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  loadCities();
  setupEventListeners();
  buildLegendPanel();
  buildStatsPanel();
});

// ── Map ───────────────────────────────────────────────────────────────────────
function initMap() {
  map = L.map('map').setView([20.5937, 78.9629], 5);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19
  }).addTo(map);
}

// ── Cities ────────────────────────────────────────────────────────────────────
async function loadCities() {
  try {
    const res  = await fetch(`${API_BASE}/api/cities`);
    const data = await res.json();
    citiesData = data.cities;

    const sel = document.getElementById('citySelect');
    sel.innerHTML = '<option value="">Select a city…</option>';

    data.cities.forEach(city => {
      const opt = document.createElement('option');
      opt.value       = city.id;
      opt.textContent = `${city.info.name}, ${city.info.country}${city.has_results ? ' ✓' : ''}`;
      sel.appendChild(opt);
    });
  } catch (err) {
    showStatus('Cannot reach API server — is it running?', 'error');
  }
}

// ── Event wiring ──────────────────────────────────────────────────────────────
function setupEventListeners() {
  document.getElementById('citySelect').addEventListener('change', e => {
    if (e.target.value) selectCity(e.target.value);
  });

  document.getElementById('analyzeBtn').addEventListener('click', runUnifiedAnalysis);

  // Layer toggle buttons (added to HTML below)
  document.getElementById('btnShowClassified')?.addEventListener('click', () =>
    showOverlay('classified'));
  document.getElementById('btnShowHeatmap')?.addEventListener('click', () =>
    showOverlay('heatmap'));
  document.getElementById('btnShowTrend')?.addEventListener('click', () =>
    showOverlay('trend'));
  document.getElementById('btnClearOverlay')?.addEventListener('click', clearOverlay);
}

// ── City selection ────────────────────────────────────────────────────────────
function selectCity(cityId) {
  currentCity = citiesData.find(c => c.id === cityId);
  if (!currentCity) return;

  const { center, bbox } = currentCity.info;
  map.setView([center.lat, center.lon], 11);

  if (cityRect) map.removeLayer(cityRect);
  cityRect = L.rectangle(
    [[bbox.south, bbox.west], [bbox.north, bbox.east]],
    { color: '#667eea', weight: 2, fillOpacity: 0.08 }
  ).addTo(map);

  showStatus(`📍 ${currentCity.info.name} selected. Choose dates and click Analyse.`, 'info');
}

// ── Core: run unified analysis ────────────────────────────────────────────────
async function runUnifiedAnalysis() {
  if (!currentCity) { showStatus('Please select a city first.', 'error'); return; }

  const beforeDate = document.getElementById('beforeDate').value;
  const afterDate  = document.getElementById('afterDate').value;
  if (!beforeDate || !afterDate) {
    showStatus('Please choose both before and after dates.', 'error');
    return;
  }

  showSpinner(true);
  setAnalyseBtn(false);
  clearOverlay();
  hideStatsPanel();
  showStatus('⏳ Fetching satellite data and running analysis…', 'info');

  try {
    const bbox = currentCity.info.bbox;

    const res = await fetch(`${API_BASE}/api/ai/analyze-changes`, {
      method:  'POST',
      headers: authHeaders(),
      body: JSON.stringify({
        bbox,
        before_date:      beforeDate,
        after_date:       afterDate,
        pixel_resolution: 10.0
      })
    });

    if (res.status === 401) {
      showStatus('Session expired — please log in again.', 'error');
      showSpinner(false); setAnalyseBtn(true);
      return;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Analysis failed');
    }

    lastResults = await res.json();
    renderResults(lastResults, bbox);
    showStatus('✅ Analysis complete!', 'success');

  } catch (err) {
    console.error(err);
    showStatus(`❌ ${err.message}`, 'error');
  } finally {
    showSpinner(false);
    setAnalyseBtn(true);
  }
}

// ── Render everything after a successful response ─────────────────────────────
function renderResults(data, bbox) {
  // 1) Draw the classified overlay on the map (most useful for users)
  const layers = data.leaflet_layers || {};
  const classifiedLayer = layers.classified_change_map;

  if (classifiedLayer?.image) {
    drawImageOverlay(classifiedLayer.image, bbox, classifiedLayer.opacity ?? 0.82);
    // Store all three images for the toggle buttons
    window._overlayImages = {
      classified: { src: classifiedLayer.image,                                         opacity: 0.82 },
      heatmap:    { src: layers.change_probability_heatmap?.image,                      opacity: 0.75 },
      trend:      { src: layers.temporal_trend_visualization?.image,                    opacity: 0.75 },
    };
    showLayerControls(true);
  }

  // 2) Update the legend to show only categories that have pixels
  updateLegend(data.classified_changes || {});

  // 3) Fill the stats panel
  fillStatsPanel(data, bbox);
}

// ── Overlay helpers ───────────────────────────────────────────────────────────

// activeOverlayKey tracks which layer is currently shown
let activeOverlayKey = 'classified';

function drawImageOverlay(base64src, bbox, opacity) {
  if (overlayLayer) map.removeLayer(overlayLayer);

  const bounds = L.latLngBounds(
    [bbox.south, bbox.west],
    [bbox.north, bbox.east]
  );

  overlayLayer = L.imageOverlay(base64src, bounds, {
    opacity,
    interactive: false,
    zIndex: 400
  }).addTo(map);

  map.fitBounds(bounds, { padding: [30, 30] });
}

function showOverlay(key) {
  if (!window._overlayImages) return;
  const layer = window._overlayImages[key];
  if (!layer?.src) { showStatus(`No ${key} layer available.`, 'error'); return; }

  activeOverlayKey = key;
  drawImageOverlay(layer.src, currentCity.info.bbox, layer.opacity);

  // Highlight active button
  ['btnShowClassified', 'btnShowHeatmap', 'btnShowTrend'].forEach(id => {
    document.getElementById(id)?.classList.remove('btn-active');
  });
  const btnMap = { classified: 'btnShowClassified', heatmap: 'btnShowHeatmap', trend: 'btnShowTrend' };
  document.getElementById(btnMap[key])?.classList.add('btn-active');
}

function clearOverlay() {
  if (overlayLayer) { map.removeLayer(overlayLayer); overlayLayer = null; }
  window._overlayImages = null;
  showLayerControls(false);
}

// ── Legend panel ──────────────────────────────────────────────────────────────
function buildLegendPanel() {
  // Insert legend HTML into the page — place wherever suits your layout
  const panel = document.getElementById('legendPanel');
  if (!panel) return;

  panel.innerHTML = `
    <h4 style="margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#555">
      Map Legend
    </h4>
    <ul id="legendList" style="list-style:none;margin:0;padding:0"></ul>
  `;
}

function updateLegend(classifiedChanges) {
  const list = document.getElementById('legendList');
  if (!list) return;
  list.innerHTML = '';

  LEGEND.forEach(({ key, label, emoji, color }) => {
    const info    = classifiedChanges[key];
    const pixels  = info?.pixels ?? 0;
    const percent = info?.percent ?? 0;

    // Dim entries with zero detections so non-technical users focus on what changed
    const opacity = pixels > 0 ? '1' : '0.35';

    const li = document.createElement('li');
    li.style.cssText = `
      display:flex; align-items:center; gap:8px;
      margin-bottom:6px; opacity:${opacity};
    `;
    li.innerHTML = `
      <span style="
        display:inline-block; width:14px; height:14px; border-radius:3px;
        background:${color}; flex-shrink:0; border:1px solid rgba(0,0,0,.15);
      "></span>
      <span style="font-size:13px; color:#333">
        ${emoji} ${label}
        ${pixels > 0 ? `<span style="color:#888;font-size:11px">(${percent}%)</span>` : ''}
      </span>
    `;
    list.appendChild(li);
  });

  // Show legend panel
  const panel = document.getElementById('legendPanel');
  if (panel) panel.style.display = 'block';
}

// ── Stats panel ───────────────────────────────────────────────────────────────
function buildStatsPanel() {
  const panel = document.getElementById('statsPanel');
  if (!panel) return;
  panel.style.display = 'none';
}

function fillStatsPanel(data, bbox) {
  const panel = document.getElementById('statsPanel');
  if (!panel) return;

  const summary   = data.change_summary   || {};
  const changes   = data.classified_changes || {};
  const trend     = data.trend_summary    || {};

  // Build rows for detected categories only
  const rows = LEGEND
    .filter(({ key }) => (changes[key]?.pixels ?? 0) > 0)
    .map(({ key, label, emoji, color }) => {
      const info       = changes[key];
      const hectares   = ((info.pixels * 100) / 1e4).toFixed(1);   // 10m pixel → 100 m²
      return `
        <tr>
          <td>
            <span style="
              display:inline-block;width:10px;height:10px;
              border-radius:2px;background:${color};margin-right:5px;
            "></span>
            ${emoji} ${label}
          </td>
          <td style="text-align:right">${info.pixels.toLocaleString()}</td>
          <td style="text-align:right">${info.percent}%</td>
          <td style="text-align:right">${hectares} ha</td>
        </tr>`;
    }).join('');

  const yearsUsed  = (data.years_used || []).join(', ');
  const cloudInfo  = Object.entries(data.cloud_percent_by_year || {})
    .map(([y, c]) => `${y}: ${c}%`).join(' · ');

  panel.innerHTML = `
    <h4 style="margin:0 0 10px;font-size:13px;text-transform:uppercase;
                letter-spacing:.05em;color:#555">
      Analysis Summary
    </h4>

    <!-- Key numbers -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px">
      ${statBox('Total Changed', `${summary.change_percent ?? 0}%`, '#667eea')}
      ${statBox('Changed Area', `${summary.change_area_hectares ?? 0} ha`, '#28B446')}
      ${statBox('Changed Pixels', (summary.changed_pixels ?? 0).toLocaleString(), '#FF8C00')}
      ${statBox('Years Analysed', yearsUsed || '—', '#555')}
    </div>

    <!-- Per-category table (only if any changes found) -->
    ${rows ? `
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead>
        <tr style="color:#888;text-transform:uppercase;font-size:10px">
          <th style="text-align:left;padding:3px 0">Type</th>
          <th style="text-align:right">Pixels</th>
          <th style="text-align:right">%</th>
          <th style="text-align:right">Area</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>` : '<p style="color:#888;font-size:12px">No significant changes detected.</p>'}

    <!-- Trend info -->
    ${trend.urban_expansion_pixels > 0 || trend.gradual_deforestation_pixels > 0 ? `
    <div style="margin-top:10px;padding:8px;background:#fff8e7;border-radius:6px;font-size:12px">
      <strong>📈 Long-term Trends</strong><br>
      ${trend.urban_expansion_pixels > 0
        ? `🏙️ Urban expansion trend: ${trend.urban_expansion_pixels.toLocaleString()} px<br>` : ''}
      ${trend.gradual_deforestation_pixels > 0
        ? `🌲 Gradual deforestation trend: ${trend.gradual_deforestation_pixels.toLocaleString()} px` : ''}
    </div>` : ''}

    <!-- Cloud info -->
    ${cloudInfo ? `
    <p style="margin:8px 0 0;font-size:10px;color:#aaa">
      ☁️ Cloud cover — ${cloudInfo}
    </p>` : ''}

    <!-- Layer toggle buttons -->
    <div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">
      <button id="btnShowClassified" onclick="showOverlay('classified')"
        class="layer-btn btn-active" title="Show categorised change map">
        🗺️ Change Map
      </button>
      <button id="btnShowHeatmap" onclick="showOverlay('heatmap')"
        class="layer-btn" title="Show change probability heatmap">
        🌡️ Heatmap
      </button>
      <button id="btnShowTrend" onclick="showOverlay('trend')"
        class="layer-btn" title="Show long-term trend layer">
        📈 Trend
      </button>
      <button id="btnClearOverlay" onclick="clearOverlay()"
        class="layer-btn" title="Remove overlay from map">
        ✖ Clear
      </button>
    </div>
  `;

  panel.style.display = 'block';

  // Re-wire buttons (they were just created by innerHTML)
  document.getElementById('btnShowClassified')?.addEventListener('click', () => showOverlay('classified'));
  document.getElementById('btnShowHeatmap')?.addEventListener('click',    () => showOverlay('heatmap'));
  document.getElementById('btnShowTrend')?.addEventListener('click',      () => showOverlay('trend'));
  document.getElementById('btnClearOverlay')?.addEventListener('click',   clearOverlay);
}

function statBox(label, value, color) {
  return `
    <div style="
      background:#f8f9fa; border-radius:8px; padding:8px 10px;
      border-left:3px solid ${color};
    ">
      <div style="font-size:10px;text-transform:uppercase;color:#888;margin-bottom:2px">
        ${label}
      </div>
      <div style="font-size:16px;font-weight:700;color:#333">${value}</div>
    </div>`;
}

function hideStatsPanel() {
  const panel = document.getElementById('statsPanel');
  if (panel) panel.style.display = 'none';
}

function showLayerControls(visible) {
  // Controls live inside statsPanel now, nothing extra needed
}

// ── Utility ───────────────────────────────────────────────────────────────────
function showStatus(message, type = 'info') {
  const el = document.getElementById('statusMessage');
  if (!el) return;
  el.innerHTML = `<div class="status-message status-${type}">${message}</div>`;
  if (type === 'success') setTimeout(() => { el.innerHTML = ''; }, 6000);
}

function showSpinner(visible) {
  const el = document.getElementById('spinner');
  if (!el) return;
  el.classList.toggle('show', visible);
}

function setAnalyseBtn(enabled) {
  const btn = document.getElementById('analyzeBtn');
  if (btn) btn.disabled = !enabled;
}

console.log('🛰️  Satellite Change Detection app loaded');