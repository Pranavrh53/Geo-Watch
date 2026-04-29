# ═══════════════════════════════════════════════════════════════════════
# GEO-WATCH: XGBoost Land Cover Classification on Sentinel-2 Data
# Copy-paste into Google Colab and run.
#
# CROSS-TEMPORAL APPROACH (correct ML):
#   Features: temporal stats from PAST years (mean, std, slope, delta)
#   Labels:   land cover class from the LATEST year
#   → The model learns temporal patterns to predict current land cover.
# ═══════════════════════════════════════════════════════════════════════

# !pip install xgboost requests numpy matplotlib scikit-learn seaborn Pillow

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests, io
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    accuracy_score, roc_curve, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
COPERNICUS_USERNAME = "frozenflames677@gmail.com"
COPERNICUS_PASSWORD = "Pranavrh123$"

BBOX = {"west": 77.3700, "south": 12.7340, "east": 77.8800, "north": 13.1730}
YEARS = list(range(2018, 2026))
IMAGE_SIZE = 512
CLASS_NAMES = ["Vegetation", "Urban", "Water", "Barren"]
CLASS_COLORS = ["#2E7D32", "#E65100", "#1565C0", "#795548"]
GOOD_SCL = {2, 4, 5, 6, 7, 11}

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

YEARLY_INDICES_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08", "B11", "SCL"], units: "DN" }],
    output: { bands: 4, sampleType: "UINT8" }
  };
}
function evaluatePixel(sample) {
  let nir = sample.B08 / 10000.0, red = sample.B04 / 10000.0;
  let swir = sample.B11 / 10000.0, green = sample.B03 / 10000.0;
  let ndvi = (nir+red)>0.001 ? (nir-red)/(nir+red) : 0.0;
  let ndbi = (swir+nir)>0.001 ? (swir-nir)/(swir+nir) : 0.0;
  let ndwi = (green+nir)>0.001 ? (green-nir)/(green+nir) : 0.0;
  return [
    Math.min(255,Math.max(0,Math.round((ndvi+1.0)*127.5))),
    Math.min(255,Math.max(0,Math.round((ndbi+1.0)*127.5))),
    Math.min(255,Math.max(0,Math.round((ndwi+1.0)*127.5))),
    Math.min(255,Math.max(0,Math.round(sample.SCL)))
  ];
}
"""

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Auth + Fetch
# ═══════════════════════════════════════════════════════════════════════
def get_access_token():
    r = requests.post(TOKEN_URL, data={
        "grant_type": "password", "username": COPERNICUS_USERNAME,
        "password": COPERNICUS_PASSWORD, "client_id": "cdse-public",
    }, timeout=60)
    r.raise_for_status()
    print("✓ Authentication successful")
    return r.json()["access_token"]

def fetch_year_data(token, year):
    body = {
        "input": {
            "bounds": {"bbox": [BBOX["west"],BBOX["south"],BBOX["east"],BBOX["north"]],
                       "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {
                "timeRange": {"from": f"{year}-01-01T00:00:00Z", "to": f"{year}-12-31T23:59:59Z"},
                "maxCloudCoverage": 95, "mosaickingOrder": "leastCC"}}],
        },
        "output": {"width": IMAGE_SIZE, "height": IMAGE_SIZE,
                   "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
        "evalscript": YEARLY_INDICES_EVALSCRIPT,
    }
    print(f"  Fetching {year}...", end=" ")
    r = requests.post(PROCESS_API_URL, json=body, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        "Accept": "image/png"}, timeout=120)
    r.raise_for_status()
    arr = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
    ndvi = (arr[:,:,0].astype(np.float32)/127.5) - 1.0
    ndbi = (arr[:,:,1].astype(np.float32)/127.5) - 1.0
    ndwi = (arr[:,:,2].astype(np.float32)/127.5) - 1.0
    scl = arr[:,:,3].astype(np.uint8)
    valid = np.isin(scl, list(GOOD_SCL))
    print(f"OK — NDVI={np.nanmean(ndvi[valid]):.4f}")
    return {"year": year, "ndvi": ndvi, "ndbi": ndbi, "ndwi": ndwi, "valid": valid}

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Ground-truth labels (from LATEST year only)
# ═══════════════════════════════════════════════════════════════════════
def generate_labels(ndvi, ndbi, ndwi):
    labels = np.full(ndvi.shape, 3, dtype=np.int32)
    labels[(ndwi > 0.0) & (ndvi < 0.25)] = 2
    labels[(ndbi > 0.0) & (ndvi < 0.25) & (labels != 2)] = 1
    labels[ndvi > 0.3] = 0
    return labels

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Build temporal features from PAST years
# ═══════════════════════════════════════════════════════════════════════
FEATURE_NAMES = [
    "NDVI_mean","NDVI_std","NDVI_min","NDVI_max","NDVI_slope","NDVI_delta",
    "NDBI_mean","NDBI_std","NDBI_min","NDBI_max","NDBI_slope","NDBI_delta",
    "NDWI_mean","NDWI_std","NDWI_min","NDWI_max","NDWI_slope","NDWI_delta",
]

def compute_temporal_features(past_data):
    T = len(past_data)
    H, W = past_data[0]["ndvi"].shape
    N = H * W
    features = []
    for key in ["ndvi", "ndbi", "ndwi"]:
        stack = np.stack([d[key] for d in past_data], axis=0)
        valid = np.stack([d["valid"] for d in past_data], axis=0)
        masked = np.where(valid, stack, np.nan).reshape(T, N)
        with np.errstate(all='ignore'):
            mn = np.nanmean(masked, axis=0)
            sd = np.nanstd(masked, axis=0)
            mi = np.nanmin(masked, axis=0)
            mx = np.nanmax(masked, axis=0)
            x = np.arange(T, dtype=np.float32)
            xm = x.mean()
            xv = np.sum((x - xm)**2)
            sl = np.nansum((x[:,None]-xm)*(masked-mn[None,:]),axis=0)/max(xv,1e-6) if xv>0 else np.zeros(N)
            dl = masked[-1] - masked[0]
        for a in [mn,sd,mi,mx,sl,dl]: a[np.isnan(a)] = 0.0
        features.extend([mn, sd, mi, mx, sl, dl])
    return np.column_stack(features).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  GEO-WATCH: XGBoost Cross-Temporal LULC Classification")
    print(f"  Features from: {YEARS[0]}–{YEARS[-2]} | Labels from: {YEARS[-1]}")
    print("=" * 60)

    token = get_access_token()
    print(f"\nFetching Sentinel-2 data for {len(YEARS)} years...")
    all_data = [fetch_year_data(token, y) for y in YEARS]

    past_data = all_data[:-1]
    target = all_data[-1]

    print(f"\n→ Building 18 temporal features from {len(past_data)} past years...")
    X_all = compute_temporal_features(past_data)
    y_all = generate_labels(target["ndvi"], target["ndbi"], target["ndwi"]).ravel()
    valid_target = target["valid"].ravel()
    past_valid_count = np.sum(np.stack([d["valid"] for d in past_data],axis=0).reshape(len(past_data),-1), axis=0)
    usable = valid_target & (past_valid_count >= 2)
    X, y = X_all[usable], y_all[usable]

    print(f"  Total usable pixels: {len(X):,}")
    for i, n in enumerate(CLASS_NAMES):
        print(f"    {n}: {np.sum(y==i):,} ({np.mean(y==i)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n→ Training XGBoost (150 trees, depth=5, 18 features)...")
    model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
        num_class=4, eval_metric="mlogloss", random_state=42, use_label_encoder=False)
    model.fit(X_train, y_train, verbose=False)
    print("✓ Training complete")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # ── ALL 8 METRICS ──
    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])
    acc = accuracy_score(y_test, y_pred)
    err = 1.0 - acc
    f1_pc = f1_score(y_test, y_pred, average=None, labels=[0,1,2,3], zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    pr_pc = precision_score(y_test, y_pred, average=None, labels=[0,1,2,3], zero_division=0)
    pr_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rc_pc = recall_score(y_test, y_pred, average=None, labels=[0,1,2,3], zero_division=0)
    rc_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    y_bin = label_binarize(y_test, classes=[0,1,2,3])
    auc_pc = [roc_auc_score(y_bin[:,i], y_prob[:,i]) if len(np.unique(y_bin[:,i]))>1 else 0 for i in range(4)]
    auc_w = np.mean(auc_pc)

    print("\n" + "="*72)
    print("  XGBOOST CROSS-TEMPORAL CLASSIFICATION METRICS")
    print("="*72)
    print(f"{'Metric':<20} {'Vegetation':>12} {'Urban':>12} {'Water':>12} {'Barren':>12} {'Weighted':>12}")
    print("-"*84)
    for lbl,vals,wt in [("Precision",pr_pc,pr_w),("Recall",rc_pc,rc_w),
                         ("Sensitivity",rc_pc,rc_w),("F1-Score",f1_pc,f1_w),("AUC",auc_pc,auc_w)]:
        print(f"{lbl:<20} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f} {vals[3]:>12.4f} {wt:>12.4f}")
    print(f"\n{'Accuracy':<20} {acc:>12.4f}")
    print(f"{'Error Rate':<20} {err:>12.4f}")
    print("="*72)

    # ══════════════════════════════════════════════════════════
    # PLOT 1: Confusion Matrix
    # ══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'1. Confusion Matrix\nAccuracy: {acc:.4f} | Error Rate: {err:.4f}', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig('1_confusion_matrix.png', dpi=150, bbox_inches='tight'); plt.show()

    # ══════════════════════════════════════════════════════════
    # PLOT 2: Precision, Recall/Sensitivity, F1 per class
    # ══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    x_pos = np.arange(4)
    for ax, vals, title in [(axes[0],pr_pc,'3. Precision'),(axes[1],rc_pc,'4. Recall / 5. Sensitivity'),(axes[2],f1_pc,'2. F1-Score')]:
        bars = ax.bar(x_pos, vals, color=[CLASS_COLORS[i] for i in range(4)], edgecolor='white', linewidth=1.2)
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.4f}', ha='center', fontweight='bold')
        ax.set_xticks(x_pos); ax.set_xticklabels(CLASS_NAMES); ax.set_ylim(0,1.15)
        ax.set_title(title, fontsize=12, fontweight='bold'); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Per-Class Metrics (Cross-Temporal)', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig('2_precision_recall_f1.png', dpi=150, bbox_inches='tight'); plt.show()

    # ══════════════════════════════════════════════════════════
    # PLOT 3: ROC + AUC
    # ══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8,7))
    for i in range(4):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
        ax.plot(fpr, tpr, color=CLASS_COLORS[i], linewidth=2, label=f'{CLASS_NAMES[i]} (AUC={auc_pc[i]:.4f})')
    ax.plot([0,1],[0,1],'k--',alpha=0.4); ax.set_xlabel('FPR',fontweight='bold'); ax.set_ylabel('TPR',fontweight='bold')
    ax.set_title(f'7. ROC Curves — AUC\nWeighted AUC = {auc_w:.4f}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('3_roc_auc.png', dpi=150, bbox_inches='tight'); plt.show()

    # ══════════════════════════════════════════════════════════
    # PLOT 4: Accuracy + Error Rate
    # ══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    axes[0].bar(['Accuracy'],[acc],color='#43A047',width=0.5)
    axes[0].text(0,acc+0.02,f'{acc:.4f}',ha='center',fontsize=16,fontweight='bold')
    axes[0].set_ylim(0,1.15); axes[0].set_title('6. Accuracy',fontweight='bold')
    axes[1].bar(['Error Rate'],[err],color='#E53935',width=0.5)
    axes[1].text(0,err+0.005,f'{err:.4f}',ha='center',fontsize=16,fontweight='bold')
    axes[1].set_ylim(0,max(0.3,err*2)); axes[1].set_title('8. Error Rate',fontweight='bold')
    plt.tight_layout(); plt.savefig('4_accuracy_error.png', dpi=150, bbox_inches='tight'); plt.show()

    # ══════════════════════════════════════════════════════════
    # PLOT 5: Feature Importance (grouped by index)
    # ══════════════════════════════════════════════════════════
    imp = model.feature_importances_
    grouped = {}
    for idx, fn in enumerate(FEATURE_NAMES):
        base = fn.split("_")[0]
        grouped[base] = grouped.get(base, 0) + imp[idx]
    total = sum(grouped.values()) or 1
    grouped = {k: v/total for k,v in grouped.items()}

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,5))
    # Grouped
    ax1.bar(grouped.keys(), grouped.values(), color=['#2E7D32','#E65100','#1565C0'], width=0.5)
    for i,(k,v) in enumerate(grouped.items()):
        ax1.text(i, v+0.01, f'{v*100:.1f}%', ha='center', fontweight='bold', fontsize=12)
    ax1.set_title('Feature Importance (Grouped)', fontweight='bold'); ax1.set_ylabel('Importance')
    # Detailed
    ax2.barh(FEATURE_NAMES, imp, color=['#2E7D32']*6+['#E65100']*6+['#1565C0']*6)
    ax2.set_title('Detailed Feature Importance (18 features)', fontweight='bold')
    ax2.invert_yaxis()
    plt.tight_layout(); plt.savefig('5_feature_importance.png', dpi=150, bbox_inches='tight'); plt.show()

    # ══════════════════════════════════════════════════════════
    # PLOT 6: Classification Map
    # ══════════════════════════════════════════════════════════
    pred_full = model.predict(X_all).reshape(IMAGE_SIZE, IMAGE_SIZE)
    gt = generate_labels(target["ndvi"], target["ndbi"], target["ndwi"])
    rgb_colors = [(46,125,50),(230,81,0),(21,101,192),(121,85,72)]

    fig, axes = plt.subplots(1,2,figsize=(14,6))
    for ax, data, title in [(axes[0],gt,'Ground Truth'),(axes[1],pred_full,'XGBoost Prediction')]:
        cmap = np.zeros((IMAGE_SIZE,IMAGE_SIZE,3), dtype=np.uint8)
        for c,rgb in enumerate(rgb_colors): cmap[data==c] = rgb
        ax.imshow(cmap); ax.set_title(f'{title} — {YEARS[-1]}', fontweight='bold'); ax.axis('off')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(CLASS_COLORS,CLASS_NAMES)],
               loc='lower center', ncol=4, fontsize=11)
    plt.suptitle('LULC Classification Map (Bangalore)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0.06,1,1]); plt.savefig('6_classification_map.png', dpi=150, bbox_inches='tight'); plt.show()

    # ══════════════════════════════════════════════════════════
    # PLOT 7: Summary Table
    # ══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10,5)); ax.axis('off')
    tbl = [['1. Confusion Matrix','See Plot 1','—'],['2. F1-Score',f'{f1_w:.4f}','Weighted'],
           ['3. Precision',f'{pr_w:.4f}','Weighted'],['4. Recall',f'{rc_w:.4f}','Weighted'],
           ['5. Sensitivity',f'{rc_w:.4f}','= Recall'],['6. Accuracy',f'{acc:.4f}',f'{acc*100:.2f}%'],
           ['7. AUC',f'{auc_w:.4f}','Mean OvR'],['8. Error Rate',f'{err:.4f}',f'{err*100:.2f}%']]
    t = ax.table(cellText=tbl, colLabels=['Metric','Value','Note'], cellLoc='center', loc='center', colWidths=[0.35,0.25,0.25])
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1,1.8)
    for j in range(3): t[0,j].set_facecolor('#1565C0'); t[0,j].set_text_props(color='white',fontweight='bold')
    for i in range(1,9):
        for j in range(3): t[i,j].set_facecolor('#f5f5f5' if i%2==0 else '#ffffff')
    ax.set_title('All 8 Metrics Summary — Cross-Temporal XGBoost', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout(); plt.savefig('7_metrics_summary.png', dpi=150, bbox_inches='tight'); plt.show()

    print("\n" + "="*60)
    print("  ALL DONE! 7 plots saved.")
    print("="*60)

if __name__ == "__main__":
    main()
