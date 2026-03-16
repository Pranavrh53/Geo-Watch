// Configuration
const API_URL = 'http://localhost:8000';
let map = null;
let drawnItems = null;
let selectedBbox = null;
let mapLayerControl = null;
let unifiedLayerRefs = {};
let accessToken = localStorage.getItem('access_token');
let currentUser = localStorage.getItem('username');

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
        document.getElementById('fetch-btn').disabled = false;
        document.getElementById('analyze-map-btn').disabled = false;
    });

    map.on(L.Draw.Event.DELETED, function () {
        selectedBbox = null;
        document.getElementById('coordinates-display').classList.remove('show');
        document.getElementById('fetch-btn').disabled = true;
        document.getElementById('analyze-map-btn').disabled = true;
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
    document.getElementById('fetch-btn').disabled = false;
    document.getElementById('analyze-map-btn').disabled = false;
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

                navigator.geolocation.getCurrentPosition(
                    function(position) {
                        const lat = position.coords.latitude;
                        const lon = position.coords.longitude;
                        
                        // Center map on user location
                        map.setView([lat, lon], 13);
                        
                        // Add a temporary marker
                        const marker = L.marker([lat, lon], {
                            icon: L.divIcon({
                                className: 'user-location-marker',
                                html: '🔵',
                                iconSize: [20, 20]
                            })
                        }).addTo(map);
                        
                        // Remove marker after 3 seconds
                        setTimeout(() => {
                            map.removeLayer(marker);
                        }, 3000);
                        
                        container.classList.remove('active');
                        
                        showAlert(`Location found: ${lat.toFixed(4)}, ${lon.toFixed(4)}`, 'success');
                    },
                    function(error) {
                        container.classList.remove('active');
                        let message = 'Unable to retrieve your location';
                        
                        switch(error.code) {
                            case error.PERMISSION_DENIED:
                                message = 'Location permission denied. Please enable location access.';
                                break;
                            case error.POSITION_UNAVAILABLE:
                                message = 'Location information unavailable.';
                                break;
                            case error.TIMEOUT:
                                message = 'Location request timed out.';
                                break;
                        }
                        
                        showAlert(message, 'error');
                    },
                    {
                        enableHighAccuracy: true,
                        timeout: 5000,
                        maximumAge: 0
                    }
                );
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
        const beforeImage = await fetchTile(selectedBbox, beforeDate);

        // Fetch after image
        document.getElementById('loading-text').textContent = 'Fetching after image (Sentinel-2)...';
        const afterImage = await fetchTile(selectedBbox, afterDate);

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
            source: beforeImage.source || 'sentinel'
        });
        window.location.href = `compare.html?${params.toString()}`;

    } catch (error) {
        console.error('Fetch error:', error);
        hideLoading();
        showAlert(error.message || 'Failed to fetch images', 'error');
        document.getElementById('fetch-btn').disabled = false;
    }
}

async function fetchTile(bbox, date) {
    const response = await fetch(`${API_URL}/api/tile/fetch`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            bbox: bbox,
            date: date,
            size: 512
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
    document.getElementById('fetch-btn').disabled = false;
    document.getElementById('analyze-map-btn').disabled = false;
}

function showLoading(text) {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading').classList.add('show');
}

function hideLoading() {
    document.getElementById('loading').classList.remove('show');
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
