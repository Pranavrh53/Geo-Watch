// Satellite Change Detection - Frontend Application

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global variables
let map;
let currentCity = null;
let citiesData = null;

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    loadCities();
    setupEventListeners();
});

// Initialize Leaflet map
function initMap() {
    // Create map centered on India
    map = L.map('map').setView([20.5937, 78.9629], 5);
    
    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);
    
    console.log('Map initialized');
}

// Load available cities from API
async function loadCities() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/cities`);
        const data = await response.json();
        
        citiesData = data.cities;
        
        // Populate city dropdown
        const citySelect = document.getElementById('citySelect');
        citySelect.innerHTML = '<option value="">Select a city</option>';
        
        data.cities.forEach(city => {
            const option = document.createElement('option');
            option.value = city.id;
            option.textContent = `${city.info.name}, ${city.info.country}`;
            if (city.has_results) {
                option.textContent += ' ✓';
            }
            citySelect.appendChild(option);
        });
        
        console.log(`Loaded ${data.cities.length} cities`);
    } catch (error) {
        console.error('Error loading cities:', error);
        showStatus('Error loading cities. Is the API server running?', 'error');
    }
}

// Setup event listeners
function setupEventListeners() {
    // City selection change
    document.getElementById('citySelect').addEventListener('change', (e) => {
        const cityId = e.target.value;
        if (cityId) {
            selectCity(cityId);
        }
    });
    
    // Analyze button
    document.getElementById('analyzeBtn').addEventListener('click', () => {
        analyzeChanges();
    });
    
    // Load results button
    document.getElementById('loadResultsBtn').addEventListener('click', () => {
        loadResults();
    });
}

// Select a city and zoom to it
function selectCity(cityId) {
    const city = citiesData.find(c => c.id === cityId);
    if (!city) return;
    
    currentCity = city;
    
    // Zoom to city
    const center = city.info.center;
    map.setView([center.lat, center.lon], 11);
    
    // Draw bounding box
    const bbox = city.info.bbox;
    const bounds = [
        [bbox.south, bbox.west],
        [bbox.north, bbox.east]
    ];
    
    // Remove previous rectangle if exists
    map.eachLayer(layer => {
        if (layer instanceof L.Rectangle) {
            map.removeLayer(layer);
        }
    });
    
    // Add rectangle
    L.rectangle(bounds, {
        color: '#667eea',
        weight: 2,
        fillOpacity: 0.1
    }).addTo(map);
    
    console.log(`Selected city: ${city.info.name}`);
    showStatus(`Selected ${city.info.name}. Choose dates and click Analyze.`, 'info');
}

// Analyze changes
async function analyzeChanges() {
    if (!currentCity) {
        showStatus('Please select a city first', 'error');
        return;
    }
    
    const beforeDate = document.getElementById('beforeDate').value;
    const afterDate = document.getElementById('afterDate').value;
    
    if (!beforeDate || !afterDate) {
        showStatus('Please select both before and after dates', 'error');
        return;
    }
    
    // Show loading
    showSpinner(true);
    document.getElementById('analyzeBtn').disabled = true;
    showStatus('Analysis started. This may take several minutes...', 'info');
    
    try {
        // Trigger analysis
        const response = await fetch(`${API_BASE_URL}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                city: currentCity.id,
                before_date: beforeDate,
                after_date: afterDate,
                run_download: false,
                run_preprocessing: false,
                run_segmentation: false,
                run_change_detection: true
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Poll for status
            pollTaskStatus(data.task_id);
        } else {
            throw new Error(data.detail || 'Analysis failed');
        }
    } catch (error) {
        console.error('Error starting analysis:', error);
        showStatus(`Error: ${error.message}`, 'error');
        showSpinner(false);
        document.getElementById('analyzeBtn').disabled = false;
    }
}

// Poll task status
async function pollTaskStatus(taskId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/status/${taskId}`);
        const status = await response.json();
        
        showStatus(`Status: ${status.status} - ${status.current_step || ''} (${status.progress}%)`, 'info');
        
        if (status.status === 'completed') {
            showStatus('✅ Analysis complete! Loading results...', 'success');
            showSpinner(false);
            document.getElementById('analyzeBtn').disabled = false;
            
            // Load results
            setTimeout(() => loadResults(), 1000);
        } else if (status.status === 'failed') {
            showStatus(`❌ Analysis failed: ${status.error}`, 'error');
            showSpinner(false);
            document.getElementById('analyzeBtn').disabled = false;
        } else {
            // Continue polling
            setTimeout(() => pollTaskStatus(taskId), 3000);
        }
    } catch (error) {
        console.error('Error polling status:', error);
        // Continue polling
        setTimeout(() => pollTaskStatus(taskId), 5000);
    }
}

// Load results for selected city
async function loadResults() {
    if (!currentCity) {
        showStatus('Please select a city first', 'error');
        return;
    }
    
    const beforeDate = document.getElementById('beforeDate').value;
    const afterDate = document.getElementById('afterDate').value;
    
    showSpinner(true);
    showStatus('Loading results...', 'info');
    
    try {
        let url = `${API_BASE_URL}/api/results/${currentCity.id}`;
        
        // Add date filters if provided
        const params = new URLSearchParams();
        if (beforeDate) params.append('before_date', beforeDate);
        if (afterDate) params.append('after_date', afterDate);
        
        if (params.toString()) {
            url += `?${params.toString()}`;
        }
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error('Results not found. Run analysis first.');
        }
        
        const results = await response.json();
        
        // Display results
        displayResults(results);
        showStatus('✅ Results loaded successfully', 'success');
        
    } catch (error) {
        console.error('Error loading results:', error);
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        showSpinner(false);
    }
}

// Display results in sidebar
function displayResults(results) {
    const resultsPanel = document.getElementById('resultsPanel');
    const resultsContent = document.getElementById('resultsContent');
    
    resultsContent.innerHTML = '';
    
    if (!results.changes || results.changes.length === 0) {
        resultsContent.innerHTML = '<p style="color: #666;">No significant changes detected.</p>';
        resultsPanel.classList.add('show');
        return;
    }
    
    // Display each change type
    results.changes.forEach(change => {
        const item = document.createElement('div');
        item.className = 'change-item';
        
        const color = `rgb(${change.color.join(',')})`;
        
        item.innerHTML = `
            <div class="change-color" style="background: ${color};"></div>
            <div class="change-info">
                <div class="change-name">${change.name}</div>
                <div class="change-area">
                    ${change.area_hectares.toFixed(2)} hectares
                    (${change.area_acres.toFixed(2)} acres)
                </div>
            </div>
        `;
        
        resultsContent.appendChild(item);
    });
    
    // Add total
    const total = document.createElement('div');
    total.style.cssText = 'margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd; font-weight: 600;';
    total.textContent = `Total Changed Area: ${results.total_change_hectares.toFixed(2)} hectares`;
    resultsContent.appendChild(total);
    
    // Show panel
    resultsPanel.classList.add('show');
    
    console.log('Results displayed:', results);
}

// Show status message
function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('statusMessage');
    statusDiv.innerHTML = `<div class="status-message status-${type}">${message}</div>`;
    
    // Auto-hide success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            statusDiv.innerHTML = '';
        }, 5000);
    }
}

// Show/hide spinner
function showSpinner(show) {
    const spinner = document.getElementById('spinner');
    if (show) {
        spinner.classList.add('show');
    } else {
        spinner.classList.remove('show');
    }
}

// Helper function to format numbers
function formatNumber(num, decimals = 2) {
    return num.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

console.log('Application initialized');
