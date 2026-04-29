# ═══════════════════════════════════════════════════════════════════════
# GEO-WATCH: Linear Regression on Sentinel-2 Spectral Indices
# Copy-paste this entire file into a Google Colab cell and run it.
#
# This is the EXACT regression logic from:
#   unified_change_detector.py → _compute_trajectory() (line 1528)
#   unified_change_detector.py → _slope_over_time() (line 969)
#
# It fetches REAL Sentinel-2 data from Copernicus, computes NDVI/NDBI/NDWI,
# then runs linear regression and outputs all metrics + plots.
# ═══════════════════════════════════════════════════════════════════════

# ── Cell 1: Install dependencies ──
# !pip install requests numpy matplotlib scikit-learn seaborn Pillow

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — Replace with your Copernicus credentials
# Get free account at: https://dataspace.copernicus.eu
# ═══════════════════════════════════════════════════════════════════════

COPERNICUS_USERNAME = "YOUR_EMAIL@example.com"   # ← Replace
COPERNICUS_PASSWORD = "YOUR_PASSWORD"             # ← Replace

# Bangalore bounding box (same as config.py)
BBOX = {
    "west": 77.3700,
    "south": 12.7340,
    "east": 77.8800,
    "north": 13.1730,
}

# Years to analyze
YEARS = list(range(2018, 2026))  # 2018–2025

# Image size (same as your project: 512×512)
IMAGE_SIZE = 512

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Authenticate with Copernicus
# Exact same logic as unified_change_detector.py line 194–218
# ═══════════════════════════════════════════════════════════════════════

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Valid SCL classes — same as line 41
GOOD_SCL = {2, 4, 5, 6, 7, 11}


def get_access_token():
    """Get Copernicus access token — same as _get_access_token()."""
    payload = {
        "grant_type": "password",
        "username": COPERNICUS_USERNAME,
        "password": COPERNICUS_PASSWORD,
        "client_id": "cdse-public",
    }
    response = requests.post(TOKEN_URL, data=payload, timeout=60)
    response.raise_for_status()
    token = response.json().get("access_token")
    print(f"✓ Authentication successful")
    return token


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Evalscript for NDVI, NDBI, NDWI
# EXACT COPY from unified_change_detector.py line 46–72
# ═══════════════════════════════════════════════════════════════════════

YEARLY_INDICES_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08", "B11", "SCL"], units: "DN" }],
    output: { bands: 4, sampleType: "UINT8" }
  };
}

function evaluatePixel(sample) {
  let blue = sample.B02 / 10000.0;
  let green = sample.B03 / 10000.0;
  let red = sample.B04 / 10000.0;
  let nir = sample.B08 / 10000.0;
  let swir = sample.B11 / 10000.0;

  let ndvi = (nir + red) > 0.001 ? (nir - red) / (nir + red) : 0.0;
  let ndbi = (swir + nir) > 0.001 ? (swir - nir) / (swir + nir) : 0.0;
  let ndwi = (green + nir) > 0.001 ? (green - nir) / (green + nir) : 0.0;

  let ndvi_u8 = Math.min(255, Math.max(0, Math.round((ndvi + 1.0) * 127.5)));
  let ndbi_u8 = Math.min(255, Math.max(0, Math.round((ndbi + 1.0) * 127.5)));
  let ndwi_u8 = Math.min(255, Math.max(0, Math.round((ndwi + 1.0) * 127.5)));
  let scl = Math.min(255, Math.max(0, Math.round(sample.SCL)));

  return [ndvi_u8, ndbi_u8, ndwi_u8, scl];
}
"""


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Fetch yearly spectral data
# Exact same logic as _fetch_year_data() line 281–455
# ═══════════════════════════════════════════════════════════════════════

def fetch_year_data(token, bbox, year, size):
    """Fetch Sentinel-2 spectral indices for one year."""
    body = {
        "input": {
            "bounds": {
                "bbox": [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{year}-01-01T00:00:00Z",
                        "to": f"{year}-12-31T23:59:59Z",
                    },
                    "maxCloudCoverage": 95,
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": size,
            "height": size,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": YEARLY_INDICES_EVALSCRIPT,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/png",
    }

    print(f"  Fetching {year}...", end=" ")
    response = requests.post(PROCESS_API_URL, json=body, headers=headers, timeout=120)
    response.raise_for_status()

    # Decode PNG — same as line 377–405
    image = Image.open(io.BytesIO(response.content)).convert("RGBA")
    arr = np.array(image)

    # Decode UINT8 back to float [-1, 1] — same as line 399–405
    ndvi = (arr[:, :, 0].astype(np.float32) / 127.5) - 1.0
    ndbi = (arr[:, :, 1].astype(np.float32) / 127.5) - 1.0
    ndwi = (arr[:, :, 2].astype(np.float32) / 127.5) - 1.0
    scl = arr[:, :, 3].astype(np.uint8)

    # Cloud masking — same as line 408
    valid = np.isin(scl, list(GOOD_SCL))
    cloud_pct = float((1.0 - np.mean(valid)) * 100.0)

    # Compute means over valid pixels only — same as line 1257
    ndvi_mean = float(np.nanmean(ndvi[valid])) if np.any(valid) else float(np.nanmean(ndvi))
    ndbi_mean = float(np.nanmean(ndbi[valid])) if np.any(valid) else float(np.nanmean(ndbi))
    ndwi_mean = float(np.nanmean(ndwi[valid])) if np.any(valid) else float(np.nanmean(ndwi))

    print(f"OK — NDVI={ndvi_mean:.4f}, NDBI={ndbi_mean:.4f}, NDWI={ndwi_mean:.4f}, Cloud={cloud_pct:.1f}%")

    return {
        "year": year,
        "ndvi_mean": round(ndvi_mean, 4),
        "ndbi_mean": round(ndbi_mean, 4),
        "ndwi_mean": round(ndwi_mean, 4),
        "cloud_percent": round(cloud_pct, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Linear Regression
# EXACT COPY from _compute_trajectory() → _regress() line 1570–1602
# ═══════════════════════════════════════════════════════════════════════

def run_regression(data_years, values, name):
    """
    Exact same regression as unified_change_detector.py line 1570–1602.
    Uses np.polyfit (degree=1) for OLS linear regression.
    """
    x = np.array(data_years, dtype=np.float64)
    y_arr = np.array(values, dtype=np.float64)

    # ── np.polyfit: Ordinary Least Squares ──
    # This fits y = slope*x + intercept by minimizing Σ(yᵢ - ŷᵢ)²
    # Same as line 1572
    coeffs = np.polyfit(x, y_arr, 1)  # returns [slope, intercept]
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # ── Predictions ──
    # Same as line 1577
    y_pred = np.polyval(coeffs, x)

    # ── R² calculation ──
    # Same as line 1578–1580
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)

    # ── Additional regression metrics ──
    mse = mean_squared_error(y_arr, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_arr, y_pred)

    # ── Projected values (10 years forward) ──
    # Same as line 1589–1592
    last_year = int(x[-1])
    projected_years = list(range(last_year + 1, last_year + 11))
    projected_values = [round(float(np.polyval(coeffs, fy)), 4) for fy in projected_years]

    # ── Trend classification ──
    # Same as line 1601
    if slope < -0.002:
        trend = "declining"
    elif slope > 0.002:
        trend = "increasing"
    else:
        trend = "stable"

    return {
        "name": name,
        "slope_per_year": round(slope, 6),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "mse": round(mse, 10),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "trend": trend,
        "y_actual": y_arr,
        "y_pred": y_pred,
        "projected_years": projected_years,
        "projected_values": projected_values,
    }


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: MAIN — Fetch data, run regression, plot everything
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  GEO-WATCH: Linear Regression on Sentinel-2 Data")
    print("  Location: Bangalore, India")
    print(f"  Years: {YEARS[0]}–{YEARS[-1]}")
    print("=" * 60)

    # ── Authenticate ──
    token = get_access_token()

    # ── Fetch real spectral data for each year ──
    print(f"\nFetching Sentinel-2 data for {len(YEARS)} years...")
    yearly_stats = []
    for year in YEARS:
        stats = fetch_year_data(token, BBOX, year, IMAGE_SIZE)
        yearly_stats.append(stats)

    # ── Extract arrays ──
    data_years = [s["year"] for s in yearly_stats]
    ndvi_vals = [s["ndvi_mean"] for s in yearly_stats]
    ndbi_vals = [s["ndbi_mean"] for s in yearly_stats]
    ndwi_vals = [s["ndwi_mean"] for s in yearly_stats]

    print(f"\n{'Year':<8} {'NDVI':>10} {'NDBI':>10} {'NDWI':>10} {'Cloud%':>10}")
    print("-" * 52)
    for s in yearly_stats:
        print(f"{s['year']:<8} {s['ndvi_mean']:>10.4f} {s['ndbi_mean']:>10.4f} "
              f"{s['ndwi_mean']:>10.4f} {s['cloud_percent']:>9.1f}%")

    # ── Run linear regression on each index ──
    ndvi_reg = run_regression(data_years, ndvi_vals, "NDVI")
    ndbi_reg = run_regression(data_years, ndbi_vals, "NDBI")
    ndwi_reg = run_regression(data_years, ndwi_vals, "NDWI")

    # ═══════════════════════════════════════════
    # PRINT ALL METRICS
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("         LINEAR REGRESSION METRICS (from real Sentinel-2 data)")
    print("=" * 70)
    print(f"{'Metric':<20} {'NDVI':>15} {'NDBI':>15} {'NDWI':>15}")
    print("-" * 70)
    for metric_name, key in [
        ("Slope (per year)", "slope_per_year"),
        ("Intercept", "intercept"),
        ("R²", "r_squared"),
        ("MSE", "mse"),
        ("RMSE", "rmse"),
        ("MAE", "mae"),
        ("Trend", "trend"),
    ]:
        v1 = ndvi_reg[key]
        v2 = ndbi_reg[key]
        v3 = ndwi_reg[key]
        if isinstance(v1, str):
            print(f"{metric_name:<20} {v1:>15} {v2:>15} {v3:>15}")
        else:
            print(f"{metric_name:<20} {v1:>15.6f} {v2:>15.6f} {v3:>15.6f}")
    print("=" * 70)

    # ═══════════════════════════════════════════
    # PLOT 1: Regression Lines with R²
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, reg, color in [
        (axes[0], ndvi_reg, '#2E7D32'),
        (axes[1], ndbi_reg, '#E65100'),
        (axes[2], ndwi_reg, '#1565C0'),
    ]:
        name = reg["name"]
        ax.scatter(data_years, reg["y_actual"], color=color, s=80,
                   zorder=3, label=f'Actual ({name})')
        ax.plot(data_years, reg["y_pred"], '--', color=color, linewidth=2,
                label=f'Linear Fit')

        # Plot projected future
        all_future_x = reg["projected_years"][:5]
        all_future_y = reg["projected_values"][:5]
        ax.plot(all_future_x, all_future_y, ':', color=color, linewidth=1.5,
                alpha=0.5, label='Projected')
        ax.scatter(all_future_x, all_future_y, color=color, s=30,
                   alpha=0.4, marker='x')

        # Residual lines
        for yr, a, p in zip(data_years, reg["y_actual"], reg["y_pred"]):
            ax.plot([yr, yr], [a, p], 'r-', alpha=0.3, linewidth=1)

        ax.set_title(f'{name} Temporal Trend\n'
                     f'R² = {reg["r_squared"]:.4f} | '
                     f'Slope = {reg["slope_per_year"]:.5f}/yr | '
                     f'Trend: {reg["trend"]}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{name} Value')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Linear Regression on Sentinel-2 Spectral Indices (Real Data)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('1_regression_trends.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 1_regression_trends.png")

    # ═══════════════════════════════════════════
    # PLOT 2: Predicted vs Actual (scatter)
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, reg, color in [
        (axes[0], ndvi_reg, '#2E7D32'),
        (axes[1], ndbi_reg, '#E65100'),
        (axes[2], ndwi_reg, '#1565C0'),
    ]:
        name = reg["name"]
        actual = reg["y_actual"]
        pred = reg["y_pred"]
        ax.scatter(actual, pred, color=color, s=80, zorder=3)
        lims = [min(min(actual), min(pred)) - 0.01,
                max(max(actual), max(pred)) + 0.01]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Fit')
        ax.set_title(f'{name}: Predicted vs Actual\nR² = {reg["r_squared"]:.4f}',
                     fontweight='bold')
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Predicted Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Predicted vs Actual — Regression Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('2_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 2_predicted_vs_actual.png")

    # ═══════════════════════════════════════════
    # PLOT 3: Residuals
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, reg, color in [
        (axes[0], ndvi_reg, '#2E7D32'),
        (axes[1], ndbi_reg, '#E65100'),
        (axes[2], ndwi_reg, '#1565C0'),
    ]:
        residuals = reg["y_actual"] - reg["y_pred"]
        ax.bar(data_years, residuals, color=color, alpha=0.7, width=0.6)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_title(f'{reg["name"]} Residuals\nMAE = {reg["mae"]:.6f}',
                     fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Residual (Actual − Predicted)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('3_residuals.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 3_residuals.png")

    # ═══════════════════════════════════════════
    # PLOT 4: Metrics Summary Bar Chart
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² comparison
    r2_vals = [ndvi_reg["r_squared"], ndbi_reg["r_squared"], ndwi_reg["r_squared"]]
    colors = ['#2E7D32', '#E65100', '#1565C0']
    bars = axes[0].bar(['NDVI', 'NDBI', 'NDWI'], r2_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars, r2_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.4f}', ha='center', fontweight='bold', fontsize=12)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title('R² (Coefficient of Determination)', fontweight='bold')
    axes[0].set_ylabel('R² Value')
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    axes[0].grid(True, alpha=0.2, axis='y')

    # RMSE comparison
    rmse_vals = [ndvi_reg["rmse"], ndbi_reg["rmse"], ndwi_reg["rmse"]]
    bars = axes[1].bar(['NDVI', 'NDBI', 'NDWI'], rmse_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                     f'{val:.6f}', ha='center', fontweight='bold', fontsize=11)
    axes[1].set_title('RMSE (Root Mean Square Error)', fontweight='bold')
    axes[1].set_ylabel('RMSE Value')
    axes[1].grid(True, alpha=0.2, axis='y')

    plt.suptitle('Regression Metrics Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('4_metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 4_metrics_summary.png")

    # ═══════════════════════════════════════════
    # PLOT 5: Trajectory Predictions (10 years)
    # ═══════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 6))

    for reg, color, marker in [
        (ndvi_reg, '#2E7D32', 'o'),
        (ndbi_reg, '#E65100', 's'),
        (ndwi_reg, '#1565C0', '^'),
    ]:
        name = reg["name"]
        # Historical
        ax.plot(data_years, reg["y_actual"], f'{marker}-', color=color,
                linewidth=2, markersize=8, label=f'{name} (actual)')
        # Projected
        ax.plot(reg["projected_years"], reg["projected_values"], f'{marker}:',
                color=color, linewidth=1.5, markersize=6, alpha=0.5,
                label=f'{name} (projected)')

    # Threshold lines
    ax.axhline(y=0.10, color='red', linestyle='--', alpha=0.4,
               label='NDVI critical (0.10)')
    ax.axhline(y=0.30, color='orange', linestyle='--', alpha=0.4,
               label='NDBI urbanized (0.30)')

    ax.axvspan(data_years[-1] + 0.5, reg["projected_years"][-1] + 0.5,
               alpha=0.05, color='gray', label='Projection zone')

    ax.set_title('10-Year Trajectory Prediction (Linear Extrapolation)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Index Value')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('5_trajectory_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 5_trajectory_prediction.png")

    print("\n" + "=" * 60)
    print("  ALL DONE! Screenshots saved as PNG files.")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
