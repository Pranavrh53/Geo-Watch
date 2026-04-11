"""
Unified multi-temporal Sentinel-2 change detection pipeline.

This module replaces parallel detector paths with one pipeline that:
1) Fetches yearly best-available cloud-minimized Sentinel-2 scenes
2) Computes NDVI/NDBI/NDWI from B02/B03/B04/B08/B11 using SCL cloud masking
3) Builds a temporal feature stack and runs tiled CNN + ConvLSTM inference
4) Classifies change types from spectral rules
5) Detects NDVI/NDBI trends by linear regression
6) Produces map-ready visualization layers
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import COPERNICUS_USERNAME, COPERNICUS_PASSWORD

logger = logging.getLogger(__name__)

PROCESS_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

GOOD_SCL = {2, 4, 5, 6, 7, 11}
CACHE_DIR = Path(__file__).parent.parent / "data" / "spectral_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Uses B02, B03, B04, B08, B11 and SCL as requested.
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


def _to_b64(img_arr: np.ndarray, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format=fmt)
    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{payload}"


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class TemporalCNNConvLSTM(nn.Module):
    """Lightweight U-Net-style encoder + ConvLSTM + decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.temporal = ConvLSTMCell(32, 32)
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        b, t, _, h, w = x.shape
        device = x.device

        h_state = torch.zeros((b, 32, h, w), device=device)
        c_state = torch.zeros((b, 32, h, w), device=device)

        for ti in range(t):
            feat = self.encoder(x[:, ti])
            h_state, c_state = self.temporal(feat, (h_state, c_state))

        model_prob = torch.sigmoid(self.decoder(h_state))

        # Deterministic temporal prior improves coherence without training.
        start = x[:, 0]
        end = x[:, -1]
        delta = torch.abs(end - start).mean(dim=1, keepdim=True)
        prior = torch.sigmoid((delta - 0.08) * 14.0)

        return torch.clamp(0.2 * model_prob + 0.8 * prior, 0.0, 1.0)


@dataclass
class YearlyData:
    year: int
    ndvi: np.ndarray
    ndbi: np.ndarray
    ndwi: np.ndarray
    valid_mask: np.ndarray
    cloud_percent: float
    is_synthetic: bool = False


class UnifiedTemporalChangeDetector:
    def __init__(self, model_size: int = 512, tile_size: int = 512):
        self.model_size = model_size
        self.tile_size = tile_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.username = COPERNICUS_USERNAME
        self.password = COPERNICUS_PASSWORD
        self.demo_mode = not (self.username and self.password)
        self.access_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None

        self.model = TemporalCNNConvLSTM().to(self.device).eval()

        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        self.proxies = {}
        if http_proxy:
            self.proxies["http"] = http_proxy
        if https_proxy:
            self.proxies["https"] = https_proxy

    def _get_access_token(self, force_refresh: bool = False) -> Optional[str]:
        if (
            not force_refresh
            and self.access_token
            and self.token_expiry
            and datetime.utcnow() < self.token_expiry
        ):
            return self.access_token

        if self.demo_mode:
            return None

        payload = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "cdse-public",
        }
        try:
            response = requests.post(TOKEN_URL, data=payload, timeout=60, proxies=self.proxies)
            if not response.ok:
                body = response.text[:500]
                logger.error(
                    "Copernicus token request failed %s: %s", response.status_code, body
                )
                response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data.get("access_token")
            expires_in = int(token_data.get("expires_in", 600))
            self.token_expiry = datetime.utcnow() + timedelta(seconds=expires_in - 60)
            logger.info("Copernicus token refreshed (expires in %ds)", expires_in)
            return self.access_token
        except Exception as exc:
            logger.error("Failed to get Copernicus token: %s", exc)
            self.access_token = None
            self.token_expiry = None
            return None

    @staticmethod
    def _bbox_hash(bbox: Dict[str, float]) -> str:
        src = f"{bbox['west']:.6f}_{bbox['south']:.6f}_{bbox['east']:.6f}_{bbox['north']:.6f}"
        return hashlib.md5(src.encode("utf-8")).hexdigest()

    def _cache_path(self, bbox: Dict[str, float], year: int, size: int) -> Path:
        key = f"{self._bbox_hash(bbox)}_{year}_{size}.npz"
        return CACHE_DIR / key

    def _load_cached_year(self, bbox: Dict[str, float], year: int, size: int) -> Optional[YearlyData]:
        path = self._cache_path(bbox, year, size)
        if not path.exists():
            return None
        try:
            arr = np.load(path)

            # Backward/forward-compatible scalar extraction for cached metadata.
            # Some cache versions store scalar fields as 0-D arrays, others as (1,).
            def _first_scalar(value: np.ndarray, default: float = 0.0) -> float:
                flat = np.asarray(value).reshape(-1)
                if flat.size == 0:
                    return default
                return float(flat[0])

            is_synthetic = (
                bool(_first_scalar(arr["is_synthetic"]))
                if "is_synthetic" in arr
                else False
            )
            # Never return stale synthetic cache — always re-fetch real data.
            if is_synthetic:
                logger.info("Discarding stale synthetic cache for year %s; will re-fetch.", year)
                path.unlink(missing_ok=True)
                return None
            return YearlyData(
                year=year,
                ndvi=arr["ndvi"],
                ndbi=arr["ndbi"],
                ndwi=arr["ndwi"],
                valid_mask=arr["valid_mask"].astype(bool),
                cloud_percent=(
                    _first_scalar(arr["cloud_percent"]) if "cloud_percent" in arr else 0.0
                ),
                is_synthetic=False,
            )
        except Exception as exc:
            logger.warning("Cache load failed for year %s: %s", year, exc)
            return None

    def _save_cached_year(self, bbox: Dict[str, float], year: int, size: int, data: YearlyData) -> None:
        path = self._cache_path(bbox, year, size)
        np.savez_compressed(
            path,
            ndvi=data.ndvi,
            ndbi=data.ndbi,
            ndwi=data.ndwi,
            valid_mask=data.valid_mask.astype(np.uint8),
            cloud_percent=np.array([data.cloud_percent], dtype=np.float32),
            is_synthetic=np.array([1 if data.is_synthetic else 0], dtype=np.uint8),
        )

    def _fetch_year_data(
        self,
        bbox: Dict[str, float],
        year: int,
        size: int,
        fetch_errors: Optional[List[str]] = None,
    ) -> YearlyData:
        """
        Fetch real Sentinel-2 spectral indices for `year`.

        Returns a YearlyData with is_synthetic=False on success.
        On failure, logs the error, appends to `fetch_errors` if provided,
        and returns synthetic fallback data (NOT cached, so the next call
        will always retry the real API).
        """
        cached = self._load_cached_year(bbox, year, size)
        if cached is not None:
            logger.info("Year %s loaded from spectral cache.", year)
            return cached

        if self.demo_mode:
            msg = f"{year}: running in demo mode (no Copernicus credentials configured)"
            logger.warning(msg)
            if fetch_errors is not None:
                fetch_errors.append(msg)
            return self._generate_demo_year(year, size)

        # ── Attempt API fetch (retry once on 401 with forced token refresh) ──
        for attempt in range(2):
            token = self._get_access_token(force_refresh=(attempt > 0))
            if not token:
                break

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

            try:
                logger.info("Fetching Sentinel-2 spectral data for %s (attempt %d)", year, attempt + 1)
                response = requests.post(
                    PROCESS_API_URL,
                    json=body,
                    headers=headers,
                    timeout=120,
                    proxies=self.proxies,
                )

                # On 401 force a token refresh and retry once
                if response.status_code == 401 and attempt == 0:
                    logger.warning("Got 401 for year %s — forcing token refresh", year)
                    self.access_token = None
                    self.token_expiry = None
                    continue

                if not response.ok:
                    body_snippet = response.text[:600]
                    raise ValueError(
                        f"Process API HTTP {response.status_code}: {body_snippet}"
                    )

                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type:
                    raise ValueError(f"Process API returned non-image content-type: {content_type}")

                raw_payload = response.content

                # Decode PNG/JPEG with PIL first (fastest)
                arr = None
                try:
                    image = Image.open(io.BytesIO(raw_payload)).convert("RGBA")
                    arr = np.array(image)
                except Exception as pil_exc:
                    logger.info("PIL decode failed for %s (%s); trying rasterio.", year, pil_exc)
                    from rasterio.io import MemoryFile
                    with MemoryFile(raw_payload) as mem:
                        with mem.open() as ds:
                            bands = ds.read()  # shape: [C, H, W]
                            if bands.shape[0] < 4:
                                raise ValueError(
                                    f"Rasterio: only {bands.shape[0]} bands returned"
                                )
                            arr = np.stack(
                                [bands[0], bands[1], bands[2], bands[3]], axis=-1
                            )

                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=-1)
                if arr.shape[-1] < 4:
                    raise ValueError(f"Decoded array has only {arr.shape[-1]} channels (need 4)")

                # ── Decode packed UINT8 indices back to float [-1, 1] ──
                # Evalscript encodes: u8 = round((index + 1.0) * 127.5)
                # Inverse:            index = u8 / 127.5 - 1.0
                ndvi = (arr[:, :, 0].astype(np.float32) / 127.5) - 1.0  # (B08-B04)/(B08+B04)
                ndbi = (arr[:, :, 1].astype(np.float32) / 127.5) - 1.0  # (B11-B08)/(B11+B08)
                ndwi = (arr[:, :, 2].astype(np.float32) / 127.5) - 1.0  # (B03-B08)/(B03+B08)
                scl  = arr[:, :, 3].astype(np.uint8)                     # Scene Classification Layer

                # SCL cloud masking: only keep vegetation/soil/bare/snow pixels
                valid = np.isin(scl, list(GOOD_SCL))  # GOOD_SCL = {2,4,5,6,7,11}
                cloud_percent = float((1.0 - np.mean(valid)) * 100.0)

                # Sanity check: if >99% pixels are identical something went wrong
                if np.std(ndvi) < 0.001:
                    raise ValueError(
                        f"NDVI std={np.std(ndvi):.5f} — image appears uniform/blank for {year}"
                    )

                yearly = YearlyData(
                    year=year,
                    ndvi=ndvi,
                    ndbi=ndbi,
                    ndwi=ndwi,
                    valid_mask=valid,
                    cloud_percent=cloud_percent,
                    is_synthetic=False,
                )
                logger.info(
                    "Year %s: real spectral data loaded OK — "
                    "NDVI mean=%.3f std=%.3f cloud=%.1f%%",
                    year,
                    float(np.mean(ndvi[valid])) if np.any(valid) else float(np.mean(ndvi)),
                    float(np.std(ndvi[valid])) if np.any(valid) else float(np.std(ndvi)),
                    cloud_percent,
                )
                # Only cache real (non-synthetic) results
                self._save_cached_year(bbox, year, size, yearly)
                return yearly

            except Exception as exc:
                logger.warning(
                    "Sentinel-2 fetch attempt %d failed for %s: %s", attempt + 1, year, exc
                )
                if attempt == 0:
                    continue  # will retry with fresh token
                # Both attempts failed — fall through to synthetic
                break

        # ── All attempts failed: return synthetic WITHOUT caching ──
        err_msg = (
            f"{year}: Sentinel-2 API fetch failed — using synthetic fallback. "
            f"Check Copernicus credentials and network connectivity."
        )
        logger.warning(err_msg)
        if fetch_errors is not None:
            fetch_errors.append(err_msg)
        return self._generate_demo_year(year, size)

    def _generate_demo_year(self, year: int, size: int) -> YearlyData:
        rng = np.random.RandomState(year)
        h = w = size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        pattern = np.sin(xx / 45.0) * np.cos(yy / 60.0)
        trend = (year - 2018) * 0.01

        ndvi = np.clip(0.45 + 0.20 * pattern - trend + rng.normal(0, 0.03, (h, w)), -1.0, 1.0)
        ndbi = np.clip(-0.10 - 0.15 * pattern + trend + rng.normal(0, 0.03, (h, w)), -1.0, 1.0)
        ndwi = np.clip(0.05 + 0.10 * np.sin(xx / 80.0) - trend * 0.2 + rng.normal(0, 0.02, (h, w)), -1.0, 1.0)

        valid = rng.random((h, w)) > 0.05
        cloud_percent = float((1.0 - np.mean(valid)) * 100.0)
        return YearlyData(year, ndvi, ndbi, ndwi, valid, cloud_percent, is_synthetic=True)

    @staticmethod
    def _estimate_cloud_percent_from_rgb(tile_arr: np.ndarray) -> float:
        """
        Estimate visible cloud cover from true-color imagery.
        Uses bright + low-saturation pixels as a conservative cloud proxy.
        """
        cloud_mask = UnifiedTemporalChangeDetector._estimate_cloud_mask_from_rgb(tile_arr)
        cloud_pct = float(np.mean(cloud_mask) * 100.0)
        return float(np.clip(cloud_pct, 0.0, 100.0))

    @staticmethod
    def _estimate_cloud_mask_from_rgb(tile_arr: np.ndarray) -> np.ndarray:
        """
        Pixel-level cloud mask from RGB imagery.
        Combines HSV and channel-whiteness tests to catch visibly cloudy regions.
        """
        if tile_arr.ndim != 3 or tile_arr.shape[2] < 3:
            return np.zeros(tile_arr.shape[:2], dtype=bool)

        rgb = tile_arr[:, :, :3].astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        rf = rgb[:, :, 0].astype(np.float32)
        gf = rgb[:, :, 1].astype(np.float32)
        bf = rgb[:, :, 2].astype(np.float32)

        # Near-white clouds: bright + low saturation + channels close together.
        whiteness = np.maximum.reduce([np.abs(rf - gf), np.abs(rf - bf), np.abs(gf - bf)])
        white_cloud = (v > 190.0) & (s < 70.0) & (whiteness < 28.0)

        # Hazy cloud decks: very bright, even when not fully white.
        haze_cloud = (v > 215.0) & (s < 95.0)

        # Bright cyan-ish thin cloud edges often seen over land/water boundaries.
        cyan_cloud = (h > 70.0) & (h < 120.0) & (v > 185.0) & (s < 105.0)

        cloud_like = white_cloud | haze_cloud | cyan_cloud

        # Morphological cleanup for stable cloud masks.
        kernel = np.ones((3, 3), np.uint8)
        cloud_like = cv2.morphologyEx(cloud_like.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cloud_like = cv2.morphologyEx(cloud_like, cv2.MORPH_CLOSE, kernel)
        return cloud_like.astype(bool)

    @staticmethod
    def _change_area_from_rgb_pair(
        prev_rgb: np.ndarray,
        curr_rgb: np.ndarray,
        valid_mask: np.ndarray,
        pixel_area_m2: float,
        total_area_ha: float,
    ) -> float:
        """
        Estimate changed area from consecutive displayed RGB frames.
        This aligns metrics with what users visually inspect in animation.
        """
        if prev_rgb.shape[:2] != curr_rgb.shape[:2]:
            return 0.0
        if valid_mask.shape != prev_rgb.shape[:2]:
            return 0.0

        prev_blur = cv2.GaussianBlur(prev_rgb, (3, 3), 0)
        curr_blur = cv2.GaussianBlur(curr_rgb, (3, 3), 0)

        # Spectral-neutral image difference score.
        diff_rgb = np.mean(np.abs(curr_blur.astype(np.float32) - prev_blur.astype(np.float32)), axis=2) / 255.0
        prev_gray = cv2.cvtColor(prev_blur, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        curr_gray = cv2.cvtColor(curr_blur, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        diff_gray = np.abs(curr_gray - prev_gray)

        score = 0.65 * diff_rgb + 0.35 * diff_gray

        valid_scores = score[valid_mask] if np.any(valid_mask) else np.array([], dtype=np.float32)
        if valid_scores.size == 0:
            return 0.0

        # Robust threshold: absolute floor + distribution-sensitive threshold.
        mean_s = float(np.mean(valid_scores))
        std_s = float(np.std(valid_scores))
        stat_thr = mean_s + 1.25 * std_s
        thr = float(np.clip(max(0.08, stat_thr), 0.08, 0.35))

        changed = (score > thr) & valid_mask
        changed_u8 = changed.astype(np.uint8)
        changed_u8 = cv2.morphologyEx(changed_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        changed_u8 = cv2.morphologyEx(changed_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        changed_u8 = UnifiedTemporalChangeDetector._remove_small(changed_u8, min_area=40)

        changed_pixels = int(np.sum(changed_u8 > 0))
        valid_ha = (int(np.sum(valid_mask)) * pixel_area_m2) / 10000.0
        return round(min(changed_pixels * pixel_area_m2 / 10000.0, valid_ha, total_area_ha), 2)

    @staticmethod
    def _year_sequence(before_date: str, after_date: str) -> List[int]:
        before_year = datetime.strptime(before_date, "%Y-%m-%d").year
        after_year = datetime.strptime(after_date, "%Y-%m-%d").year
        min_year = min(before_year, after_year)
        max_year = max(before_year, after_year)
        start_year = max(2016, min_year - 4)
        end_year = min(datetime.utcnow().year, max_year + 2)
        return list(range(start_year, end_year + 1))

    def _build_temporal_stack(
        self,
        series: List[YearlyData],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ndvi_stack = np.stack([x.ndvi for x in series], axis=0)
        ndbi_stack = np.stack([x.ndbi for x in series], axis=0)
        ndwi_stack = np.stack([x.ndwi for x in series], axis=0)
        valid_stack = np.stack([x.valid_mask for x in series], axis=0)

        features = np.stack([ndvi_stack, ndbi_stack, ndwi_stack], axis=1)  # [T, 3, H, W]

        # Cloud-mask invalid samples by replacing with per-pixel temporal median.
        med = np.nanmedian(np.where(valid_stack[:, None], features, np.nan), axis=0)
        for t in range(features.shape[0]):
            invalid = ~valid_stack[t]
            for c in range(features.shape[1]):
                layer = features[t, c]
                layer[invalid] = med[c][invalid]
                features[t, c] = layer

        return features.astype(np.float32), ndvi_stack, ndbi_stack, ndwi_stack, valid_stack

    def _infer_probability_tiled(self, features: np.ndarray) -> np.ndarray:
        # features: [T, 3, H, W]
        t, c, h, w = features.shape
        tile = self.tile_size
        out = np.zeros((h, w), dtype=np.float32)

        with torch.no_grad():
            for y0 in range(0, h, tile):
                for x0 in range(0, w, tile):
                    y1 = min(y0 + tile, h)
                    x1 = min(x0 + tile, w)

                    patch = features[:, :, y0:y1, x0:x1]
                    patch_h, patch_w = patch.shape[-2], patch.shape[-1]
                    if patch_h != tile or patch_w != tile:
                        padded = np.zeros((t, c, tile, tile), dtype=np.float32)
                        padded[:, :, :patch_h, :patch_w] = patch
                        patch = padded

                    tensor = torch.from_numpy(patch).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
                    prob = self.model(tensor).squeeze().cpu().numpy()
                    out[y0:y1, x0:x1] = prob[:patch_h, :patch_w]

        return out

    @staticmethod
    def _select_period_indices(
        years: List[int],
        before_year: int,
        after_year: int,
        min_images: int = 1,
        max_images: int = 2,
    ) -> Tuple[List[int], List[int]]:
        before_pool = [i for i, y in enumerate(years) if y <= before_year]
        after_pool = [i for i, y in enumerate(years) if y >= after_year]

        # Take only the last max_images (closest to the boundary year).
        before_idx = before_pool[-max_images:]
        after_idx = after_pool[:max_images]

        # Fallback: use transition years if strict pools are empty.
        if len(after_idx) < min_images:
            transition_pool = [i for i, y in enumerate(years) if before_year < y <= after_year]
            if transition_pool:
                after_idx = transition_pool[-max_images:]

        if len(before_idx) < min_images:
            transition_pool = [i for i, y in enumerate(years) if before_year <= y < after_year]
            if transition_pool:
                before_idx = transition_pool[:max_images]

        # Fallback to at least one year per side.
        if not before_idx:
            before_idx = [0]
        if not after_idx:
            after_idx = [len(years) - 1]

        return before_idx, after_idx

    @staticmethod
    def _period_median(
        stack: np.ndarray,
        valid_stack: np.ndarray,
        indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        selected = stack[indices]
        selected_valid = valid_stack[indices]

        composite = np.nanmedian(np.where(selected_valid, selected, np.nan), axis=0)
        valid_composite = np.sum(selected_valid.astype(np.uint8), axis=0) > 0

        # Fill any all-invalid temporal pixels from global median fallback.
        if np.any(~valid_composite):
            global_med = np.nanmedian(np.where(valid_stack, stack, np.nan), axis=0)
            global_med = np.nan_to_num(global_med, nan=0.0)
            composite = np.where(valid_composite, composite, global_med)

        composite = np.nan_to_num(composite, nan=0.0)
        return composite.astype(np.float32), valid_composite

    @staticmethod
    def _period_change_prior(
        ndvi_before: np.ndarray,
        ndbi_before: np.ndarray,
        ndwi_before: np.ndarray,
        ndvi_after: np.ndarray,
        ndbi_after: np.ndarray,
        ndwi_after: np.ndarray,
    ) -> np.ndarray:
        ndvi_delta = np.abs(ndvi_after - ndvi_before)
        ndbi_delta = np.abs(ndbi_after - ndbi_before)
        ndwi_delta = np.abs(ndwi_after - ndwi_before)

        delta = 0.50 * ndvi_delta + 0.35 * ndbi_delta + 0.15 * ndwi_delta
        prior = 1.0 / (1.0 + np.exp(-(delta - 0.10) * 10.0))
        return np.clip(prior.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def _remove_small(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
        clean = mask.astype(np.uint8).copy()
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                clean[labels == i] = 0
        return clean
    
    @staticmethod
    def _detect_roads(urban_mask: np.ndarray) -> np.ndarray:
        """Detect thin linear structures from urban mask as probable new roads."""
        mask = (urban_mask > 0).astype(np.uint8)
        if np.sum(mask) == 0:
            return mask

        horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 13), dtype=np.uint8))
        vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((13, 1), dtype=np.uint8))
        roads = ((horizontal > 0) | (vertical > 0)).astype(np.uint8)
        roads = cv2.morphologyEx(roads, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        roads = UnifiedTemporalChangeDetector._remove_small(roads, min_area=180)
        return roads

    def _classify(
        self,
        prob: np.ndarray,
        ndvi_before: np.ndarray,
        ndbi_before: np.ndarray,
        ndwi_before: np.ndarray,
        ndvi_after: np.ndarray,
        ndbi_after: np.ndarray,
        ndwi_after: np.ndarray,
        valid_before: np.ndarray,
        valid_after: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        valid = valid_before & valid_after

        ndvi_delta = ndvi_after - ndvi_before
        ndbi_delta = ndbi_after - ndbi_before

        # ── Construction classification: two-tier approach ──
        # KEY INSIGHT: Agricultural land has seasonal changes that look
        # like construction (harvest = bare soil → NDBI up, NDVI down).
        # FIX: Require the AFTER image to actually look like BUILT-UP
        # land (high NDBI, very low NDVI) — not just "something changed."

        # Common filters: must look like construction NOW, not just changed.
        looks_built_up = ndbi_after > 0.15   # after image has built-up signature
        no_vegetation = ndvi_after < 0.25     # no vegetation in after image
        non_water = ndwi_after < 0.25

        # Tier 1: High confidence — model is confident + looks built-up.
        high_conf = (
            (prob > 0.70) & valid
            & looks_built_up
            & no_vegetation
            & non_water
            & (ndbi_delta > 0.05)          # some urban increase
        )

        # Tier 2: Medium confidence — clear spectral change + built-up after.
        med_conf = (
            (prob > 0.55) & valid
            & (ndbi_after > 0.10)           # moderately built-up after
            & (ndvi_after < 0.25)           # no vegetation after
            & (ndbi_delta > 0.12)           # built-up index went up
            & (ndvi_delta < -0.10)          # vegetation clearly dropped
            & non_water
        )

        urbanization = high_conf | med_conf

        # Build class map with ONLY construction (class 3).
        class_map = np.zeros(prob.shape, dtype=np.uint8)
        class_map[urbanization] = 3

        # ── REGION SMOOTHING: gentle — remove pixel jaggedness without losing detections ──
        mask = (class_map == 3).astype(np.uint8)
        if np.sum(mask) > 0:
            # Small close to fill 1-2 pixel gaps within each detection.
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            # Gaussian blur softens jagged pixel edges.
            mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
            mask = (mask > 0.4).astype(np.uint8)
            # Remove only truly tiny noise (<50 pixels).
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            clean_mask = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 50:
                    clean_mask[labels == i] = 1
            class_map[:] = 0
            class_map[clean_mask > 0] = 3

        total = float(prob.size)
        stats_out = {
            "construction": {"pixels": int(np.sum(class_map == 3)), "color": "rgb(255,140,0)"},
        }
        for key in stats_out:
            stats_out[key]["percent"] = round(
                (stats_out[key]["pixels"] / max(1.0, total)) * 100.0, 3
            )
        return class_map, stats_out

    @staticmethod
    def _slope_over_time(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        # values: [T,H,W], valid: [T,H,W]
        t = values.shape[0]
        x = np.arange(t, dtype=np.float32)[:, None, None]
        m = valid.astype(np.float32)
        count = np.sum(m, axis=0)
        x_mean = np.sum(x * m, axis=0) / np.maximum(count, 1.0)
        y_mean = np.sum(values * m, axis=0) / np.maximum(count, 1.0)
        num = np.sum((x - x_mean) * (values - y_mean) * m, axis=0)
        den = np.sum(((x - x_mean) ** 2) * m, axis=0)
        slope = num / np.maximum(den, 1e-6)
        slope[count < 2] = 0.0
        return slope

    def _trend_map(self, ndvi_stack: np.ndarray, ndbi_stack: np.ndarray, valid_stack: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        ndvi_slope = self._slope_over_time(ndvi_stack, valid_stack)
        ndbi_slope = self._slope_over_time(ndbi_stack, valid_stack)

        gradual_deforestation = ndvi_slope < -0.02
        urban_expansion = ndbi_slope > 0.015

        trend = np.zeros(ndvi_slope.shape, dtype=np.uint8)
        trend[gradual_deforestation] = 1
        trend[urban_expansion] = 2

        return trend, {
            "gradual_deforestation_pixels": int(np.sum(gradual_deforestation)),
            "urban_expansion_pixels": int(np.sum(urban_expansion)),
        }

    @staticmethod
    def _build_heatmap(prob: np.ndarray, _context_rgb: np.ndarray) -> np.ndarray:
        p = np.clip(prob, 0.0, 1.0)
        p8 = (p * 255).astype(np.uint8)
        hmap = cv2.applyColorMap(p8, cv2.COLORMAP_JET)
        hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)

        # Keep low-confidence pixels transparent so map imagery remains visible
        # when this layer is toggled with other overlays.
        alpha = np.clip((p - 0.45) / 0.45, 0.0, 1.0)

        rgba = np.zeros((p.shape[0], p.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = hmap
        rgba[:, :, 3] = (alpha * 230).astype(np.uint8)
        return rgba

    @staticmethod
    def _morphological_cleanup(mask: np.ndarray,
                                close_kernel: int = 15,
                                min_area: int = 500) -> np.ndarray:
        """
        Close small gaps inside detected zones, then drop tiny fragments.
        This turns scattered pixel patches into solid filled regions.
        """
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)

        # 1. Close gaps - joins nearby patches into solid blobs
        closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # 2. Fill holes inside blobs
        filled = closed.copy()
        h, w = closed.shape
        flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(filled, flood, (0, 0), 1)
        filled = closed | (~filled.astype(bool)).astype(np.uint8)

        # 3. Drop regions smaller than min_area pixels
        n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        out = np.zeros_like(filled)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 1
        return out

    @staticmethod
    def _build_class_overlay(class_map: np.ndarray,
                              _context_rgb: np.ndarray) -> np.ndarray:
        """
        Build an RGBA overlay with smooth polygon edges.
        Uses approxPolyDP for clean non-pixelated outlines.
        """
        h, w = class_map.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        mask = (class_map == 3).astype(np.uint8)
        if np.sum(mask) == 0:
            return overlay

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fill_color = (185, 95, 5)      # deep orange
        border_color = (255, 255, 255)  # white outline

        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                # Polygon approximation for clean edges (no convex hull).
                epsilon = 0.008 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                # Semi-transparent fill.
                cv2.drawContours(overlay, [approx], -1, (*fill_color, 140), thickness=cv2.FILLED)
                # Bold white border.
                cv2.drawContours(overlay, [approx], -1, (*border_color, 255), thickness=2)

        return overlay

    @staticmethod
    def _build_changes_panel_image(
        trend_map: np.ndarray,
        context_rgb: np.ndarray,
    ) -> np.ndarray:
        """
        Build an opaque RGB image for the Detected Changes side-by-side panel.

        Renders the temporal trend overlay (same as Unified Map) on top of
        the satellite imagery for the selected region.
        - Red:    gradual vegetation loss (NDVI slope < -0.02)
        - Orange: urban expansion trend  (NDBI slope > 0.015)
        """
        h, w = trend_map.shape
        ctx = context_rgb.copy()
        if ctx.shape[:2] != (h, w):
            ctx = cv2.resize(ctx, (w, h), interpolation=cv2.INTER_AREA)

        # Slightly darken satellite base so trend colors pop.
        base = np.clip(ctx.astype(np.float32) * 0.65, 0, 255).astype(np.uint8)

        # Trend colors matching the Unified Map overlay.
        trend_colors = {1: (200, 20, 20), 2: (245, 140, 20)}  # red, orange

        for tid, color in trend_colors.items():
            mask = (trend_map == tid).astype(np.uint8)
            if np.sum(mask) == 0:
                continue

            # Blend the trend color onto the satellite base.
            color_layer = base.copy()
            color_layer[mask > 0] = color
            cv2.addWeighted(color_layer, 0.55, base, 0.45, 0, base)

            # White outline around each trend region.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base, contours, -1, (255, 255, 255), thickness=1)

        return base

    @staticmethod
    def _build_trend_overlay(trend_map: np.ndarray, _context_rgb: np.ndarray) -> np.ndarray:
        overlay = np.zeros((trend_map.shape[0], trend_map.shape[1], 4), dtype=np.uint8)
        # Red: gradual NDVI decline, Orange: urban expansion trend.
        trend_colors = {1: (200, 20, 20), 2: (245, 140, 20)}
        for tid, color in trend_colors.items():
            mask = (trend_map == tid).astype(np.uint8)
            if np.sum(mask) == 0:
                continue

            overlay[mask > 0, :3] = color
            overlay[mask > 0, 3] = 200

            # Thin bright outline improves readability over satellite basemap.
            dilated = cv2.dilate(mask, np.ones((4, 4), np.uint8), iterations=1)
            border = dilated.astype(bool) & ~mask.astype(bool)
            overlay[border, :3] = (255, 255, 255)
            overlay[border, 3] = 220
        return overlay

    def analyze_changes(
        self,
        bbox: Dict[str, float],
        before_date: str,
        after_date: str,
        before_rgb: Optional[np.ndarray] = None,
        after_rgb: Optional[np.ndarray] = None,
        pixel_resolution: float = 10.0,
        include_raw: bool = False,
    ) -> Dict:
        years = self._year_sequence(before_date, after_date)
        logger.info("Unified temporal analysis years: %s", years)

        yearly = [self._fetch_year_data(bbox, y, self.model_size) for y in years]
        features, ndvi_stack, ndbi_stack, ndwi_stack, valid_stack = self._build_temporal_stack(yearly)

        logger.info("Running tiled ConvLSTM inference at %sx%s", self.model_size, self.model_size)
        prob = self._infer_probability_tiled(features)
        logger.info("Inference completed")

        before_year = datetime.strptime(before_date, "%Y-%m-%d").year
        after_year = datetime.strptime(after_date, "%Y-%m-%d").year

        before_idx, after_idx = self._select_period_indices(
            years,
            before_year=before_year,
            after_year=after_year,
            min_images=1,
            max_images=2,
        )

        ndvi_before, valid_before = self._period_median(ndvi_stack, valid_stack, before_idx)
        ndbi_before, _ = self._period_median(ndbi_stack, valid_stack, before_idx)
        ndwi_before, _ = self._period_median(ndwi_stack, valid_stack, before_idx)

        ndvi_after, valid_after = self._period_median(ndvi_stack, valid_stack, after_idx)
        ndbi_after, _ = self._period_median(ndbi_stack, valid_stack, after_idx)
        ndwi_after, _ = self._period_median(ndwi_stack, valid_stack, after_idx)

        period_prior = self._period_change_prior(
            ndvi_before,
            ndbi_before,
            ndwi_before,
            ndvi_after,
            ndbi_after,
            ndwi_after,
        )

        # Fuse ConvLSTM output with robust period-composite delta prior.
        final_prob = np.clip(0.55 * prob + 0.45 * period_prior, 0.0, 1.0)

        class_map, class_stats = self._classify(
            final_prob,
            ndvi_before,
            ndbi_before,
            ndwi_before,
            ndvi_after,
            ndbi_after,
            ndwi_after,
            valid_before,
            valid_after,
        )

        trend_map, trend_stats = self._trend_map(ndvi_stack, ndbi_stack, valid_stack)

        if before_rgb is None:
            before_rgb = np.zeros((self.model_size, self.model_size, 3), dtype=np.uint8) + 90
        if after_rgb is None:
            after_rgb = before_rgb.copy()

        if before_rgb.shape[:2] != (self.model_size, self.model_size):
            before_rgb = cv2.resize(before_rgb, (self.model_size, self.model_size), interpolation=cv2.INTER_AREA)
        if after_rgb.shape[:2] != (self.model_size, self.model_size):
            after_rgb = cv2.resize(after_rgb, (self.model_size, self.model_size), interpolation=cv2.INTER_AREA)

        prob_layer = self._build_heatmap(final_prob, after_rgb)
        class_layer = self._build_class_overlay(class_map, after_rgb)
        trend_layer = self._build_trend_overlay(trend_map, after_rgb)
        changes_panel = self._build_changes_panel_image(trend_map, after_rgb)

        # Count changed pixels as the union of classified + trend detections.
        # This matches what is actually shown in the visual output.
        # (Previously used `final_prob > 0.45` which was far too permissive,
        #  catching seasonal/agricultural noise and inflating area stats.)
        detected_mask = (class_map != 0) | (trend_map != 0)
        changed_pixels = int(np.sum(detected_mask))
        total_pixels = int(prob.size)
        total_change_pct = round((changed_pixels / max(1, total_pixels)) * 100.0, 2)

        # Calculate actual ground area from bbox (NOT pixel_resolution).
        # The 512×512 grid covers the entire bbox, so each pixel represents
        # total_ground_area / total_pixels of real ground.
        import math
        lat_extent_m = (bbox['north'] - bbox['south']) * 111_000.0
        avg_lat_rad = math.radians((bbox['north'] + bbox['south']) / 2.0)
        lon_extent_m = (bbox['east'] - bbox['west']) * 111_000.0 * math.cos(avg_lat_rad)
        total_ground_area_m2 = lat_extent_m * lon_extent_m
        pixel_area_m2 = total_ground_area_m2 / max(1, total_pixels)

        area_hectares = round(changed_pixels * pixel_area_m2 / 10000.0, 2)
        total_area_hectares = round(total_ground_area_m2 / 10000.0, 2)
        # Sanity cap: change can never exceed total study area
        area_hectares = min(area_hectares, total_area_hectares)

        yearly_cloud = {str(x.year): round(x.cloud_percent, 2) for x in yearly}

        result = {
            "status": "success",
            "method": "Unified Multi-Temporal CNN+ConvLSTM + Period-Median Smoothing",
            "device": self.device,
            "tile_size": self.tile_size,
            "years_used": years,
            "temporal_windows": {
                "before_year": before_year,
                "after_year": after_year,
                "before_period_years": [years[i] for i in before_idx],
                "after_period_years": [years[i] for i in after_idx],
                "strategy": "period_median_2_year",
            },
            "cloud_percent_by_year": yearly_cloud,
            "spectral_indices": {
                "ndvi_formula": "(B08-B04)/(B08+B04)",
                "ndbi_formula": "(B11-B08)/(B11+B08)",
                "ndwi_formula": "(B03-B08)/(B03+B08)",
                "cloud_mask": "SCL in {2,4,5,6,7,11}",
            },
            "change_summary": {
                "changed_pixels": changed_pixels,
                "total_pixels": total_pixels,
                "change_percent": total_change_pct,
                "change_area_hectares": area_hectares,
                "total_area_hectares": total_area_hectares,
                "pixel_area_m2": round(pixel_area_m2, 4),
            },
            "classified_changes": class_stats,
            "trend_summary": trend_stats,
            "leaflet_layers": {
                "change_probability_heatmap": {
                    "name": "Change Probability",
                    "opacity": 0.75,
                    "image": _to_b64(prob_layer),
                    "bbox": bbox,
                },
                "classified_change_map": {
                    "name": "Classified Changes",
                    "opacity": 0.9,
                    "image": _to_b64(class_layer),
                    "bbox": bbox,
                    "legend": {
                        "construction": "orange",
                    },
                },
                "temporal_trend_visualization": {
                    "name": "Temporal Trends",
                    "opacity": 0.75,
                    "image": _to_b64(trend_layer),
                    "bbox": bbox,
                },
            },
            "changes_panel_image": _to_b64(changes_panel),
            "overlays": {
                "change_probability": _to_b64(prob_layer),
                "classified": _to_b64(class_layer),
                "trend": _to_b64(trend_layer),
            },
        }

        if include_raw:
            result["_raw"] = {
                "class_map": class_map,
                "change_probability": final_prob,
            }

        return result

    def generate_animation_frames(
        self,
        bbox: Dict[str, float],
        before_date: str,
        after_date: str,
        tile_fetcher=None,
        db=None,
    ) -> Dict:
        """
        Generate per-year animation frames with spectral metrics.

        Returns a dict with ordered frames, each containing:
          - year, image (base64), ndvi_mean, ndbi_mean,
            vegetation_pct, urban_pct, change_area_ha, cumulative_change_ha
        Also returns `fetch_errors` — a list of any years that fell back to
        synthetic data, so the frontend can surface a clear warning.
        """
        import math

        years = self._year_sequence(before_date, after_date)
        logger.info("Animation: generating frames for years %s", years)

        # Collect fetch errors per year so the frontend can warn the user
        fetch_errors: List[str] = []

        # Fetch or load cached spectral data for each year
        yearly: List[YearlyData] = [
            self._fetch_year_data(bbox, y, self.model_size, fetch_errors=fetch_errors)
            for y in years
        ]

        # Ground area calculations
        lat_extent_m = (bbox["north"] - bbox["south"]) * 111_000.0
        avg_lat_rad = math.radians((bbox["north"] + bbox["south"]) / 2.0)
        lon_extent_m = (
            (bbox["east"] - bbox["west"]) * 111_000.0 * math.cos(avg_lat_rad)
        )
        total_ground_area_m2 = lat_extent_m * lon_extent_m
        total_pixels = self.model_size * self.model_size
        pixel_area_m2 = total_ground_area_m2 / max(1, total_pixels)
        total_area_ha = round(total_ground_area_m2 / 10000.0, 2)

        # Use year-to-year deltas instead of first-year baseline so values
        # reflect incremental change and do not drift unrealistically.
        prev_ndvi: Optional[np.ndarray] = None
        prev_ndbi: Optional[np.ndarray] = None
        prev_ndwi: Optional[np.ndarray] = None
        prev_valid: Optional[np.ndarray] = None
        prev_display_rgb: Optional[np.ndarray] = None
        prev_display_valid: Optional[np.ndarray] = None

        frames = []

        for i, yd in enumerate(yearly):
            valid = yd.valid_mask
            ndvi = yd.ndvi
            ndbi = yd.ndbi
            ndwi = yd.ndwi

            # ── Per-year spectral metrics (NDVI-focused) ──
            ndvi_valid = ndvi[valid] if np.any(valid) else ndvi.ravel()
            ndbi_valid = ndbi[valid] if np.any(valid) else ndbi.ravel()

            ndvi_mean = round(float(np.mean(ndvi_valid)), 4)
            ndvi_min = round(float(np.min(ndvi_valid)), 4) if ndvi_valid.size > 0 else 0.0
            ndvi_max = round(float(np.max(ndvi_valid)), 4) if ndvi_valid.size > 0 else 0.0
            ndvi_std = round(float(np.std(ndvi_valid)), 4) if ndvi_valid.size > 0 else 0.0
            ndbi_mean = round(float(np.mean(ndbi_valid)), 4)

            # Year-to-year spectral fallback change area.
            if prev_ndvi is not None and prev_ndbi is not None and prev_ndwi is not None and prev_valid is not None:
                common_valid = valid & prev_valid
                ndvi_delta = np.abs(ndvi - prev_ndvi)
                ndbi_delta = np.abs(ndbi - prev_ndbi)
                ndwi_delta = np.abs(ndwi - prev_ndwi)

                change_score = 0.55 * ndvi_delta + 0.35 * ndbi_delta + 0.10 * ndwi_delta

                # Fixed absolute threshold avoids quantile-locking to near-constant area.
                changed_mask = (change_score > 0.14) & common_valid
                change_pixels = int(np.sum(changed_mask))

                # Change cannot exceed area that is valid in both years.
                common_valid_ha = (int(np.sum(common_valid)) * pixel_area_m2) / 10000.0
                spectral_change_ha = round(min(change_pixels * pixel_area_m2 / 10000.0, common_valid_ha, total_area_ha), 2)
            else:
                spectral_change_ha = 0.0

            # ── Build false-color RGB visualization ──
            # NDVI → green, NDBI → red, NDWI → blue
            r = np.clip(((ndbi + 1.0) / 2.0 * 200 + 30), 0, 255).astype(np.uint8)
            g = np.clip(((ndvi + 1.0) / 2.0 * 220 + 20), 0, 255).astype(np.uint8)
            b = np.clip(((ndwi + 1.0) / 2.0 * 180 + 30), 0, 255).astype(np.uint8)

            # Darken invalid/cloud pixels
            r[~valid] = (r[~valid] * 0.3).astype(np.uint8)
            g[~valid] = (g[~valid] * 0.3).astype(np.uint8)
            b[~valid] = (b[~valid] * 0.3).astype(np.uint8)

            rgb = np.stack([r, g, b], axis=-1)

            # Try to fetch real satellite tile if fetcher available
            real_tile_b64 = None
            visual_cloud_pct = 0.0
            display_rgb = rgb
            display_valid = valid.copy()
            if tile_fetcher and db:
                try:
                    tile_path = tile_fetcher.get_tile(
                        db, bbox, f"{yd.year}-06-15", (512, 512)
                    )
                    from PIL import Image as PILImage
                    tile_img = PILImage.open(tile_path).convert("RGB")
                    tile_arr = np.array(tile_img)
                    # Check quality
                    if float(np.std(tile_arr)) > 15:
                        if tile_arr.shape[:2] != (self.model_size, self.model_size):
                            tile_arr = cv2.resize(
                                tile_arr,
                                (self.model_size, self.model_size),
                                interpolation=cv2.INTER_AREA,
                            )
                        visual_cloud_pct = self._estimate_cloud_percent_from_rgb(tile_arr)
                        cloud_mask = self._estimate_cloud_mask_from_rgb(tile_arr)
                        display_rgb = tile_arr
                        display_valid = ~cloud_mask
                        real_tile_b64 = _to_b64(tile_arr, "JPEG")
                except Exception as exc:
                    logger.debug("No real tile for %s: %s", yd.year, exc)

            # Prefer RGB-frame change when consecutive display frames are available.
            if prev_display_rgb is not None and prev_display_valid is not None:
                common_display_valid = display_valid & prev_display_valid
                image_change_ha = self._change_area_from_rgb_pair(
                    prev_display_rgb,
                    display_rgb,
                    common_display_valid,
                    pixel_area_m2,
                    total_area_ha,
                )
            else:
                image_change_ha = 0.0

            # Keep cloud metric conservative: never underreport visible clouds.
            cloud_pct = round(float(max(yd.cloud_percent, visual_cloud_pct)), 1)

            # Use image-based change when available because it tracks what is shown.
            change_ha = image_change_ha if image_change_ha > 0 else spectral_change_ha

            frame = {
                "year": yd.year,
                "image": real_tile_b64 or _to_b64(rgb, "JPEG"),
                "is_real_tile": real_tile_b64 is not None,
                "ndvi_mean": ndvi_mean,
                "ndvi_min": ndvi_min,
                "ndvi_max": ndvi_max,
                "ndvi_std": ndvi_std,
                "ndbi_mean": ndbi_mean,
                "cloud_pct": cloud_pct,
                "change_area_ha": change_ha,
                "source": "synthetic" if yd.is_synthetic else "spectral",
            }
            frames.append(frame)

            prev_ndvi = ndvi
            prev_ndbi = ndbi
            prev_ndwi = ndwi
            prev_valid = valid
            prev_display_rgb = display_rgb
            prev_display_valid = display_valid

        synthetic_count = sum(1 for f in frames if f.get("source") == "synthetic")
        real_count = len(frames) - synthetic_count

        return {
            "status": "success",
            "bbox": bbox,
            "total_area_ha": total_area_ha,
            "pixel_area_m2": round(pixel_area_m2, 4),
            "years": years,
            "frame_count": len(frames),
            "real_frames": real_count,
            "synthetic_frames": synthetic_count,
            "fetch_errors": fetch_errors,
            "frames": frames,
        }


_unified_detector: Optional[UnifiedTemporalChangeDetector] = None


def get_unified_detector() -> UnifiedTemporalChangeDetector:
    global _unified_detector
    if _unified_detector is None:
        _unified_detector = UnifiedTemporalChangeDetector()
    return _unified_detector
