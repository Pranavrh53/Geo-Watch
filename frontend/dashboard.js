// Configuration
const API_URL = 'http://localhost:8000';
let map = null;
let drawnItems = null;
let selectedBbox = null;
let mapLayerControl = null;
let unifiedLayerRefs = {};
let knotIntervalId = null;
let knotA = 0;
let knotB = 0;
let userLocationMarker = null;
let userLocationAccuracyCircle = null;
let accessToken = localStorage.getItem('access_token');
let currentUser = localStorage.getItem('username');

const KNOT_W = 56;
const KNOT_H = 24;
const KNOT_RAMP = '.,-~:;=!*#$@';
const KNOT_COLORS = ['#00F5FF', '#FF2D95', '#F7FF33', '#FF6A00', '#9BFF00', '#FFFFFF'];
const GEO_TARGET_ACCURACY_M = 150;
const GEO_RELIABLE_ACCURACY_M = 5000;
const GEO_MAX_ACCEPTABLE_ACCURACY_M = 25000;
const GEO_LOCATE_TIMEOUT_MS = 25000;

function norm(v) {
    const mag = Math.sqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z)) || 1;
    return { x: v.x / mag, y: v.y / mag, z: v.z / mag };
}

function dot(a, b) {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

function cross(a, b) {
    return {
        x: (a.y * b.z) - (a.z * b.y),
        y: (a.z * b.x) - (a.x * b.z),
        z: (a.x * b.y) - (a.y * b.x)
    };
}

function startKnotLoader() {
    const el = document.getElementById('knot-loader');
    if (!el || knotIntervalId) return;

    const renderFrame = () => {
        const screen = new Array(KNOT_W * KNOT_H).fill(' ');
        const colorIdx = new Array(KNOT_W * KNOT_H).fill(-1);
        const zbuf = new Array(KNOT_W * KNOT_H).fill(0);

        const light = norm({ x: -1, y: 1, z: -1 });
        const cA = Math.cos(knotA);
        const sA = Math.sin(knotA);
        const cB = Math.cos(knotB);
        const sB = Math.sin(knotB);

        let tubeIdx = 0;
        for (let u = 0; u < 2 * Math.PI; u += 0.07, tubeIdx++) {
            const c2 = 2 * u;
            const c3 = 3 * u;
            const center = {
                x: Math.sin(u) + 2 * Math.sin(c2),
                y: Math.cos(u) - 2 * Math.cos(c2),
                z: -Math.sin(c3)
            };

            const tangent = norm({
                x: Math.cos(u) + 4 * Math.cos(c2),
                y: -Math.sin(u) + 4 * Math.sin(c2),
                z: -3 * Math.cos(c3)
            });

            const up = Math.abs(dot(tangent, { x: 0, y: 1, z: 0 })) < 0.99
                ? { x: 0, y: 1, z: 0 }
                : { x: 1, y: 0, z: 0 };

            const normal = norm(cross(tangent, up));
            const binormal = cross(tangent, normal);
            const radius = 0.3;
            const segColor = tubeIdx % KNOT_COLORS.length;

            for (let v = 0; v < 2 * Math.PI; v += 0.24) {
                const cv = Math.cos(v);
                const sv = Math.sin(v);
                const offset = {
                    x: normal.x * cv * radius + binormal.x * sv * radius,
                    y: normal.y * cv * radius + binormal.y * sv * radius,
                    z: normal.z * cv * radius + binormal.z * sv * radius
                };

                const point = {
                    x: center.x + offset.x,
                    y: center.y + offset.y,
                    z: center.z + offset.z
                };

                const x1 = point.x;
                const y1 = (point.y * cA) - (point.z * sA);
                const z1 = (point.y * sA) + (point.z * cA);

                const x2 = (x1 * cB) + (z1 * sB);
                const y2 = y1;
                const z2 = (-x1 * sB) + (z1 * cB) + 5;
                const invz = 1 / z2;

                const px = Math.floor((KNOT_W / 2) + (KNOT_W * 0.55 * x2 * invz));
                const py = Math.floor((KNOT_H / 2) - (KNOT_H * 0.85 * y2 * invz));

                if (px >= 0 && px < KNOT_W && py >= 0 && py < KNOT_H) {
                    const idx = px + (py * KNOT_W);
                    if (invz > zbuf[idx]) {
                        zbuf[idx] = invz;

                        const n = norm(offset);
                        const nx1 = n.x;
                        const ny1 = (n.y * cA) - (n.z * sA);
                        const nz1 = (n.y * sA) + (n.z * cA);

                        const nx2 = (nx1 * cB) + (nz1 * sB);
                        const ny2 = ny1;
                        const nz2 = (-nx1 * sB) + (nz1 * cB);

                        const lum = Math.max(0, dot({ x: nx2, y: ny2, z: nz2 }, light));
                        const ci = Math.min(KNOT_RAMP.length - 1, Math.floor(lum * (KNOT_RAMP.length - 1)));
                        screen[idx] = KNOT_RAMP[ci];
                        colorIdx[idx] = segColor;
                    }
                }
            }
        }

        let html = '';
        for (let y = 0; y < KNOT_H; y++) {
            for (let x = 0; x < KNOT_W; x++) {
                const idx = x + (y * KNOT_W);
                const ch = screen[idx];
                if (ch === ' ') {
                    html += ' ';
                } else {
                    html += `<span style="color:${KNOT_COLORS[colorIdx[idx]]}">${ch}</span>`;
                }
            }
            html += '\n';
        }

        el.innerHTML = html;
        knotA += 0.04;
        knotB += 0.024;
    };

    renderFrame();
    knotIntervalId = setInterval(renderFrame, 42);
}

function stopKnotLoader() {
    const el = document.getElementById('knot-loader');
    if (knotIntervalId) {
        clearInterval(knotIntervalId);
        knotIntervalId = null;
    }
    if (el) el.innerHTML = '';
}

const FEATURE_MODE_LABELS = {
    natural_color: 'Natural Color',
    geology: 'Geology',
    ndvi: 'NDVI',
    bathymetric: 'Bathymetric',
    infrared: 'Infrared',
    moisture_index: 'Moisture Index',
    ndwi: 'NDWI'
};

const FEATURE_SCAN_MODES = [
    { mode: 'natural_color', description: 'What you see' },
    { mode: 'geology', description: 'Rocks & soil' },
    { mode: 'ndvi', description: 'Plant health' },
    { mode: 'bathymetric', description: 'Water depth' },
    { mode: 'infrared', description: 'Vegetation detection' },
    { mode: 'moisture_index', description: 'Wet vs dry' },
    { mode: 'ndwi', description: 'Water detection' }
];

function setPrimaryActionButtonsEnabled(enabled) {
    const fetchBtn = document.getElementById('fetch-btn');
    if (fetchBtn) fetchBtn.disabled = !enabled;
    const scanAllBtn = document.getElementById('scan-all-btn');
    if (scanAllBtn) scanAllBtn.disabled = !enabled;
}

// City coordinates
const CITIES = {
    bangalore: { center: [12.9716, 77.5946], zoom: 11 },
    mumbai: { center: [19.0760, 72.8777], zoom: 11 },
    delhi: { center: [28.7041, 77.1025], zoom: 11 },
    hyderabad: { center: [17.3850, 78.4867], zoom: 11 }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Check if logged in
    if (!accessToken) {
        window.location.href = 'login.html';
        return;
    }

    // Set username
    document.getElementById('username').textContent = currentUser || 'User';

    // Initialize map
    initMap();

    // Load history
    loadHistory();
});

function initMap() {
    // Create map centered on Bangalore by default
    map = L.map('map').setView([12.9716, 77.5946], 11);

    // Define base layers
    const baseLayers = {
        "Streets": L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }),
        "Satellite": L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '© Esri, Maxar, Earthstar Geographics',
            maxZoom: 19
        }),
        "Terrain": L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenTopoMap contributors',
            maxZoom: 17
        }),
        "Dark Mode": L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '© CARTO',
            maxZoom: 19
        })
    };

    // Add default layer (Satellite)
    baseLayers["Satellite"].addTo(map);

    // Add layer control
    mapLayerControl = L.control.layers(baseLayers).addTo(map);

    // Initialize drawn items layer
    drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    // Add geolocation control
    addGeolocationControl();

    // Add drawing control
    const drawControl = new L.Control.Draw({
        position: 'topright',
        draw: {
            polygon: false,
            polyline: false,
            circle: false,
            circlemarker: false,
            marker: false,
            rectangle: {
                shapeOptions: {
                    color: '#667eea',
                    weight: 3
                }
            }
        },
        edit: {
            featureGroup: drawnItems,
            remove: true
        }
    });
    map.addControl(drawControl);

    // Handle drawing events
    map.on(L.Draw.Event.CREATED, function (event) {
        const layer = event.layer;
        
        // Clear previous drawings
        drawnItems.clearLayers();
        
        // Add new drawing
        drawnItems.addLayer(layer);
        
        // Get bounds
        const bounds = layer.getBounds();
        selectedBbox = {
            north: bounds.getNorth(),
            south: bounds.getSouth(),
            east: bounds.getEast(),
            west: bounds.getWest()
        };
        
        // Display coordinates
        displayCoordinates(selectedBbox);
        
        // Enable fetch button
        setPrimaryActionButtonsEnabled(true);
        const analyzeBtn = document.getElementById('analyze-map-btn');
        if (analyzeBtn) analyzeBtn.disabled = false;
    });

    map.on(L.Draw.Event.DELETED, function () {
        selectedBbox = null;
        document.getElementById('coordinates-display').classList.remove('show');
        setPrimaryActionButtonsEnabled(false);
        const analyzeBtn = document.getElementById('analyze-map-btn');
        if (analyzeBtn) analyzeBtn.disabled = true;
    });
}

function applyManualCoords() {
    const west  = parseFloat(document.getElementById('manual-west').value);
    const south = parseFloat(document.getElementById('manual-south').value);
    const east  = parseFloat(document.getElementById('manual-east').value);
    const north = parseFloat(document.getElementById('manual-north').value);

    if ([west, south, east, north].some(isNaN)) {
        alert('Please fill in all four coordinate fields.');
        return;
    }
    if (west >= east || south >= north) {
        alert('Invalid bbox: West must be < East and South must be < North.');
        return;
    }

    // Set the global bbox
    selectedBbox = { north, south, east, west };

    // Draw rectangle on the map
    if (typeof drawnItems !== 'undefined') {
        drawnItems.clearLayers();
    }
    const bounds = L.latLngBounds([south, west], [north, east]);
    const rect = L.rectangle(bounds, {
        color: '#B2E600', weight: 3, fillOpacity: 0.15, dashArray: '8 4'
    });
    if (typeof drawnItems !== 'undefined') {
        drawnItems.addLayer(rect);
    } else {
        rect.addTo(map);
    }

    // Fly to the area
    map.fitBounds(bounds, { padding: [40, 40] });

    // Update sidebar display and enable fetch
    displayCoordinates(selectedBbox);
    setPrimaryActionButtonsEnabled(true);
    const analyzeBtn = document.getElementById('analyze-map-btn');
    if (analyzeBtn) analyzeBtn.disabled = false;
}

function clearUnifiedOverlays() {
    const keys = Object.keys(unifiedLayerRefs);
    keys.forEach((key) => {
        const layer = unifiedLayerRefs[key];
        if (layer) {
            map.removeLayer(layer);
            if (mapLayerControl) {
                mapLayerControl.removeLayer(layer);
            }
        }
    });
    unifiedLayerRefs = {};
}

async function runUnifiedAnalysisOnMap() {
    if (!selectedBbox) {
        showAlert('Select a region first.', 'error');
        return;
    }

    const beforeDate = document.getElementById('before-date').value;
    const afterDate = document.getElementById('after-date').value;
    if (!beforeDate || !afterDate) {
        showAlert('Select before and after dates.', 'error');
        return;
    }

    showLoading('Running unified multi-temporal analysis...');
    document.getElementById('analyze-map-btn').disabled = true;

    try {
        const response = await fetch(`${API_URL}/api/ai/analyze-changes`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                bbox: selectedBbox,
                before_date: beforeDate,
                after_date: afterDate,
                pixel_resolution: 10.0
            })
        });

        if (!response.ok) {
            if (response.status === 401) {
                logout();
                return;
            }
            const error = await response.json();
            throw new Error(error.detail || 'Unified analysis failed');
        }

        const result = await response.json();
        const layers = result.leaflet_layers || {};
        const bounds = L.latLngBounds(
            [selectedBbox.south, selectedBbox.west],
            [selectedBbox.north, selectedBbox.east]
        );

        clearUnifiedOverlays();

        if (layers.change_probability_heatmap?.image) {
            const layer = L.imageOverlay(
                layers.change_probability_heatmap.image,
                bounds,
                { opacity: layers.change_probability_heatmap.opacity || 0.75 }
            ).addTo(map);
            unifiedLayerRefs.probability = layer;
            if (mapLayerControl) mapLayerControl.addOverlay(layer, 'Change Probability Heatmap');
        }

        if (layers.classified_change_map?.image) {
            const layer = L.imageOverlay(
                layers.classified_change_map.image,
                bounds,
                { opacity: layers.classified_change_map.opacity || 0.8 }
            ).addTo(map);
            unifiedLayerRefs.classified = layer;
            if (mapLayerControl) mapLayerControl.addOverlay(layer, 'Classified Change Map');
        }

        if (layers.temporal_trend_visualization?.image) {
            const layer = L.imageOverlay(
                layers.temporal_trend_visualization.image,
                bounds,
                { opacity: layers.temporal_trend_visualization.opacity || 0.75 }
            ).addTo(map);
            unifiedLayerRefs.trend = layer;
            if (mapLayerControl) mapLayerControl.addOverlay(layer, 'Temporal Trend Visualization');
        }

        map.fitBounds(bounds, { padding: [20, 20] });
        showAlert('Unified map overlays added. Use layer control to toggle layers.', 'success');
    } catch (error) {
        console.error('Unified map analysis failed:', error);
        showAlert(error.message || 'Unified map analysis failed', 'error');
    } finally {
        hideLoading();
        document.getElementById('analyze-map-btn').disabled = false;
    }
}

function addGeolocationControl() {
    // Create custom geolocation control
    const GeolocationControl = L.Control.extend({
        options: {
            position: 'topleft'
        },

        onAdd: function(map) {
            const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control leaflet-control-locate');
            container.innerHTML = '📍';
            container.title = 'Find My Location';

            container.onclick = function() {
                container.classList.add('active');
                
                if (!navigator.geolocation) {
                    alert('Geolocation is not supported by your browser');
                    container.classList.remove('active');
                    return;
                }

                let bestPosition = null;
                let watchId = null;
                let settled = false;

                const placeLocationMarker = (position) => {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    const accuracyMeters = Math.max(0, position.coords.accuracy || 0);

                    const zoomLevel = accuracyMeters > 25000 ? 9 : accuracyMeters > 5000 ? 11 : 15;
                    map.flyTo([lat, lon], zoomLevel, { duration: 1.1 });

                    if (userLocationMarker && map.hasLayer(userLocationMarker)) {
                        map.removeLayer(userLocationMarker);
                    }
                    if (userLocationAccuracyCircle && map.hasLayer(userLocationAccuracyCircle)) {
                        map.removeLayer(userLocationAccuracyCircle);
                    }

                    userLocationMarker = L.marker([lat, lon], {
                        icon: L.divIcon({
                            className: 'user-location-marker',
                            html: '🔵',
                            iconSize: [20, 20],
                            iconAnchor: [10, 18],
                            popupAnchor: [0, -16]
                        })
                    }).addTo(map);

                    userLocationAccuracyCircle = L.circle([lat, lon], {
                        radius: accuracyMeters,
                        color: '#00F5FF',
                        weight: 2,
                        fillColor: '#00F5FF',
                        fillOpacity: 0.12
                    }).addTo(map);

                    userLocationMarker.bindPopup(
                        `Your Location<br>Lat: ${lat.toFixed(5)}, Lon: ${lon.toFixed(5)}<br>Accuracy: ±${Math.round(accuracyMeters)} m`
                    ).openPopup();

                    if (accuracyMeters <= GEO_RELIABLE_ACCURACY_M) {
                        localStorage.setItem('geoWatchLastReliableLocation', JSON.stringify({
                            lat,
                            lon,
                            accuracy: accuracyMeters,
                            ts: Date.now()
                        }));
                    }

                    if (accuracyMeters > GEO_MAX_ACCEPTABLE_ACCURACY_M) {
                        showAlert(`Approximate location shown (±${Math.round(accuracyMeters)}m). Turn on GPS/precise location for better accuracy.`, 'error');
                    } else {
                        showAlert(`Location found: ${lat.toFixed(4)}, ${lon.toFixed(4)} (±${Math.round(accuracyMeters)}m)`, 'success');
                    }
                };

                const finalize = () => {
                    if (settled) return;
                    settled = true;

                    if (watchId !== null) {
                        navigator.geolocation.clearWatch(watchId);
                    }

                    container.classList.remove('active');

                    if (!bestPosition) {
                        showAlert('Unable to retrieve your location.', 'error');
                        return;
                    }

                    const accuracyMeters = Math.max(0, bestPosition.coords.accuracy || 0);
                    if (accuracyMeters > GEO_MAX_ACCEPTABLE_ACCURACY_M) {
                        const lastReliableRaw = localStorage.getItem('geoWatchLastReliableLocation');
                        if (lastReliableRaw) {
                            try {
                                const lastReliable = JSON.parse(lastReliableRaw);
                                placeLocationMarker({
                                    coords: {
                                        latitude: lastReliable.lat,
                                        longitude: lastReliable.lon,
                                        accuracy: lastReliable.accuracy
                                    }
                                });
                                showAlert(`Current GPS fix is too coarse (±${Math.round(accuracyMeters)}m). Showing last reliable location instead.`, 'error');
                                return;
                            } catch (e) {
                                console.warn('Failed to parse last reliable location:', e);
                            }
                        }

                        showAlert(`Location fix is too coarse (±${Math.round(accuracyMeters)}m). Move to open sky or enable precise location, then retry.`, 'error');
                        return;
                    }

                    placeLocationMarker(bestPosition);
                };

                const fallbackLocate = () => {
                    navigator.geolocation.getCurrentPosition(
                        function(position) {
                            if (settled) return;
                            bestPosition = position;
                            finalize();
                        },
                        function() {
                            if (settled) return;
                            settled = true;
                            if (watchId !== null) {
                                navigator.geolocation.clearWatch(watchId);
                            }
                            container.classList.remove('active');
                            showAlert('Could not detect location. Check browser/site location settings and try again.', 'error');
                        },
                        {
                            enableHighAccuracy: false,
                            timeout: 8000,
                            maximumAge: 120000
                        }
                    );
                };

                watchId = navigator.geolocation.watchPosition(
                    function(position) {
                        if (!bestPosition || position.coords.accuracy < bestPosition.coords.accuracy) {
                            bestPosition = position;
                        }

                        const accuracyMeters = Math.max(0, bestPosition.coords.accuracy || 0);
                        if (accuracyMeters <= GEO_TARGET_ACCURACY_M) {
                            finalize();
                        }
                    },
                    function(error) {
                        if (settled) return;

                        if (error.code === error.PERMISSION_DENIED) {
                            settled = true;
                            if (watchId !== null) {
                                navigator.geolocation.clearWatch(watchId);
                            }
                            container.classList.remove('active');
                            showAlert('Location permission denied. Please enable location access.', 'error');
                            return;
                        }

                        if (!bestPosition) {
                            fallbackLocate();
                            return;
                        }

                        finalize();
                    },
                    {
                        enableHighAccuracy: true,
                        timeout: GEO_LOCATE_TIMEOUT_MS,
                        maximumAge: 30000
                    }
                );

                setTimeout(() => {
                    if (settled) return;
                    if (bestPosition) {
                        finalize();
                    } else {
                        fallbackLocate();
                    }
                }, GEO_LOCATE_TIMEOUT_MS);
            };

            return container;
        }
    });

    map.addControl(new GeolocationControl());
}

function displayCoordinates(bbox) {
    const display = document.getElementById('coordinates-display');
    display.innerHTML = `
        North: ${bbox.north.toFixed(4)}<br>
        South: ${bbox.south.toFixed(4)}<br>
        East: ${bbox.east.toFixed(4)}<br>
        West: ${bbox.west.toFixed(4)}<br>
        <br>
        <small>Area: ~${calculateArea(bbox).toFixed(2)} km²</small>
    `;
    display.classList.add('show');
}

function calculateArea(bbox) {
    // Rough calculation in km²
    const latDiff = bbox.north - bbox.south;
    const lonDiff = bbox.east - bbox.west;
    const avgLat = (bbox.north + bbox.south) / 2;
    
    // 1 degree latitude ≈ 111 km
    const latKm = latDiff * 111;
    const lonKm = lonDiff * 111 * Math.cos(avgLat * Math.PI / 180);
    
    return latKm * lonKm;
}

function jumpToCity() {
    const cityId = document.getElementById('city-select').value;
    if (cityId && CITIES[cityId]) {
        const city = CITIES[cityId];
        map.setView(city.center, city.zoom);
    }
}

async function fetchImages() {
    if (!selectedBbox) {
        showAlert('Please select a region on the map first', 'error');
        return;
    }

    const beforeDate = document.getElementById('before-date').value;
    const afterDate = document.getElementById('after-date').value;
    const selectedMode = 'natural_color';

    if (!beforeDate || !afterDate) {
        showAlert('Please select both dates', 'error');
        return;
    }

    // Show loading
    showLoading('Fetching satellite images...');
    document.getElementById('fetch-btn').disabled = true;

    try {
        // Fetch before image
        document.getElementById('loading-text').textContent = 'Fetching before image (Sentinel-2)...';
        const beforeImage = await fetchTile(selectedBbox, beforeDate, selectedMode);

        // Fetch after image
        document.getElementById('loading-text').textContent = 'Fetching after image (Sentinel-2)...';
        const afterImage = await fetchTile(selectedBbox, afterDate, selectedMode);

        // Check if images are empty/invalid
        const beforeEmpty = beforeImage.quality && !beforeImage.quality.is_valid;
        const afterEmpty = afterImage.quality && !afterImage.quality.is_valid;

        if (beforeEmpty || afterEmpty) {
            hideLoading();
            
            // Log quality details for debugging
            console.log('Before image quality:', beforeImage.quality);
            console.log('After image quality:', afterImage.quality);
            
            const message = beforeEmpty && afterEmpty 
                ? `Both images appear to have no satellite data for these dates.\n\nBefore: ${beforeImage.quality?.reason || 'No data'}\nAfter: ${afterImage.quality?.reason || 'No data'}\n\nTry selecting different dates or a different region.`
                : beforeEmpty 
                ? `Before image (${beforeDate}) has no satellite data.\n\nReason: ${beforeImage.quality?.reason || 'No data'}\n\nTry selecting a different date.`
                : `After image (${afterDate}) has no satellite data.\n\nReason: ${afterImage.quality?.reason || 'No data'}\n\nTry selecting a different date.`;
            
            alert(message);
            document.getElementById('fetch-btn').disabled = false;
            return;
        }

        // Log successful quality checks
        console.log('Images fetched successfully:', {
            before: beforeImage.quality,
            after: afterImage.quality
        });

        // Save to history
        await saveToHistory(selectedBbox, beforeDate, afterDate);

        // Hide loading
        hideLoading();

        // Redirect to comparison viewer
        const params = new URLSearchParams({
            before: beforeImage.image_url,
            after: afterImage.image_url,
            beforeDate: beforeDate,
            afterDate: afterDate,
            bbox: JSON.stringify(selectedBbox),
            source: beforeImage.source || 'sentinel',
            renderMode: beforeImage.render_mode || selectedMode,
            renderLabel: beforeImage.render_label || FEATURE_MODE_LABELS[selectedMode] || 'Natural Color'
        });
        window.location.href = `compare.html?${params.toString()}`;

    } catch (error) {
        console.error('Fetch error:', error);
        hideLoading();
        showAlert(error.message || 'Failed to fetch images', 'error');
        document.getElementById('fetch-btn').disabled = false;
    }
}

async function fetchTile(bbox, date, renderMode) {
    const response = await fetch(`${API_URL}/api/tile/fetch`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            bbox: bbox,
            date: date,
            size: 512,
            render_mode: renderMode || 'natural_color'
        })
    });

    if (!response.ok) {
        if (response.status === 401) {
            logout();
            throw new Error('Session expired. Please login again.');
        }
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch tile');
    }

    return await response.json();
}

async function fetchAllFeatureScans() {
    if (!selectedBbox) {
        showAlert('Please select a region on the map first', 'error');
        return;
    }

    const beforeDate = document.getElementById('before-date').value;
    const afterDate = document.getElementById('after-date').value;

    if (!beforeDate || !afterDate) {
        showAlert('Please select both dates', 'error');
        return;
    }

    showLoading('Processing all feature scans...');
    const scanBtn = document.getElementById('scan-all-btn');
    if (scanBtn) scanBtn.disabled = true;

    try {
        const results = [];

        for (let i = 0; i < FEATURE_SCAN_MODES.length; i++) {
            const entry = FEATURE_SCAN_MODES[i];
            const label = FEATURE_MODE_LABELS[entry.mode] || entry.mode;
            document.getElementById('loading-text').textContent = `Processing ${label} (${i + 1}/${FEATURE_SCAN_MODES.length})...`;

            const before = await fetchTile(selectedBbox, beforeDate, entry.mode);
            const after = await fetchTile(selectedBbox, afterDate, entry.mode);

            results.push({
                mode: entry.mode,
                label,
                description: entry.description,
                before_image_url: before.image_url,
                after_image_url: after.image_url,
                data_source: after.data_source || before.data_source || null
            });
        }

        await saveToHistory(selectedBbox, beforeDate, afterDate);

        const payload = {
            bbox: selectedBbox,
            beforeDate,
            afterDate,
            source: 'sentinel',
            scans: results,
            generatedAt: new Date().toISOString()
        };

        sessionStorage.setItem('geoWatchFeatureScans', JSON.stringify(payload));
        hideLoading();
        window.location.href = 'scan_all.html';
    } catch (error) {
        console.error('All feature scan failed:', error);
        hideLoading();
        showAlert(error.message || 'Failed to process all feature scans', 'error');
        if (scanBtn) scanBtn.disabled = false;
    }
}

async function saveToHistory(bbox, beforeDate, afterDate) {
    try {
        await fetch(`${API_URL}/api/history/save`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                bbox: bbox,
                before_date: beforeDate,
                after_date: afterDate
            })
        });
    } catch (error) {
        console.error('Failed to save history:', error);
    }
}

async function loadHistory() {
    try {
        const response = await fetch(`${API_URL}/api/history/list`, {
            headers: {
                'Authorization': `Bearer ${accessToken}`
            }
        });

        if (response.ok) {
            const history = await response.json();
            displayHistory(history);
        }
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

function displayHistory(history) {
    const container = document.getElementById('history-list');
    
    if (history.length === 0) {
        container.innerHTML = '<div style="opacity: 0.6; font-size: 0.9em;">No recent analysis</div>';
        return;
    }

    container.innerHTML = history.slice(0, 5).map(item => `
        <div class="history-item" onclick='loadHistoryItem(${JSON.stringify(item)})'>
            <div class="history-item-title">
                ${item.region_name || 'Custom Region'}
            </div>
            <div class="history-item-date">
                ${item.before_date} → ${item.after_date}
            </div>
        </div>
    `).join('');
}

function loadHistoryItem(item) {
    // Zoom to the region
    const bbox = item.bbox;
    const bounds = L.latLngBounds(
        [bbox.south, bbox.west],
        [bbox.north, bbox.east]
    );
    map.fitBounds(bounds);

    // Draw the rectangle
    drawnItems.clearLayers();
    const rectangle = L.rectangle(bounds, {
        color: '#667eea',
        weight: 3
    });
    drawnItems.addLayer(rectangle);

    // Set the bbox
    selectedBbox = bbox;
    displayCoordinates(bbox);

    // Set dates
    document.getElementById('before-date').value = item.before_date;
    document.getElementById('after-date').value = item.after_date;

    // Enable fetch button
    setPrimaryActionButtonsEnabled(true);
    const analyzeBtn = document.getElementById('analyze-map-btn');
    if (analyzeBtn) analyzeBtn.disabled = false;
}

function showLoading(text) {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading').classList.add('show');
    startKnotLoader();
}

function hideLoading() {
    document.getElementById('loading').classList.remove('show');
    stopKnotLoader();
}

function showAlert(message, type = 'error') {
    const alert = document.getElementById('alert');
    alert.textContent = message;
    alert.className = `alert ${type} show`;
    setTimeout(() => alert.classList.remove('show'), 5000);
}

function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('username');
    window.location.href = 'login.html';
}
