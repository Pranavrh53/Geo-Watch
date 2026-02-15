// Configuration
const API_URL = 'http://localhost:8000';
let map = null;
let drawnItems = null;
let selectedBbox = null;
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

    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);

    // Initialize drawn items layer
    drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

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
    });

    map.on(L.Draw.Event.DELETED, function () {
        selectedBbox = null;
        document.getElementById('coordinates-display').classList.remove('show');
        document.getElementById('fetch-btn').disabled = true;
    });
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
        document.getElementById('loading-text').textContent = 'Fetching before image...';
        const beforeImage = await fetchTile(selectedBbox, beforeDate);

        // Fetch after image
        document.getElementById('loading-text').textContent = 'Fetching after image...';
        const afterImage = await fetchTile(selectedBbox, afterDate);

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
            bbox: JSON.stringify(selectedBbox)
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
