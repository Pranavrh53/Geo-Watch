"""
Geo-Watch monitoring alerts — DB model, CRUD, and threshold evaluator.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text
)
from sqlalchemy.orm import Session

from backend.database import Base

logger = logging.getLogger(__name__)


# ── SQLAlchemy model ────────────────────────────────────────────────────────

class MonitoringAlert(Base):
    __tablename__ = "monitoring_alerts"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, index=True, nullable=False)
    name          = Column(String, nullable=False)

    # Bounding box
    bbox_west     = Column(Float, nullable=False)
    bbox_south    = Column(Float, nullable=False)
    bbox_east     = Column(Float, nullable=False)
    bbox_north    = Column(Float, nullable=False)

    # Thresholds (any None = not checked)
    thr_ndvi_drop  = Column(Float, nullable=True)   # trigger if NDVI drops > this
    thr_ndbi_rise  = Column(Float, nullable=True)   # trigger if NDBI rises > this
    thr_ndwi_change= Column(Float, nullable=True)   # trigger if |ΔNDWI| > this
    thr_area_ha    = Column(Float, nullable=True)   # trigger if changed area > this ha

    # Notification
    email         = Column(String, nullable=False)
    frequency     = Column(String, default="weekly")   # "daily" | "weekly" | "monthly"

    # Baseline (stored as JSON string)
    baseline_json = Column(Text, nullable=True)   # {"ndvi":..., "ndbi":..., "ndwi":...}

    # Runtime state
    status        = Column(String, default="active")   # "active" | "paused" | "triggered"
    last_checked  = Column(DateTime, nullable=True)
    last_triggered= Column(DateTime, nullable=True)
    trigger_count = Column(Integer, default=0)
    last_result_json = Column(Text, nullable=True)  # JSON of last check result

    created_at    = Column(DateTime, default=datetime.utcnow)


# ── Helper: bbox dict ────────────────────────────────────────────────────────

def _bbox_from_alert(alert: MonitoringAlert) -> Dict[str, float]:
    return {
        "west": alert.bbox_west,
        "south": alert.bbox_south,
        "east": alert.bbox_east,
        "north": alert.bbox_north,
    }


def _baseline(alert: MonitoringAlert) -> Dict[str, float]:
    if alert.baseline_json:
        try:
            return json.loads(alert.baseline_json)
        except Exception:
            pass
    return {}


def alert_to_dict(alert: MonitoringAlert) -> Dict[str, Any]:
    return {
        "id": alert.id,
        "user_id": alert.user_id,
        "name": alert.name,
        "bbox": _bbox_from_alert(alert),
        "thresholds": {
            "ndvi_drop":   alert.thr_ndvi_drop,
            "ndbi_rise":   alert.thr_ndbi_rise,
            "ndwi_change": alert.thr_ndwi_change,
            "area_ha":     alert.thr_area_ha,
        },
        "email":      alert.email,
        "frequency":  alert.frequency,
        "baseline":   _baseline(alert),
        "status":     alert.status,
        "last_checked":   alert.last_checked.isoformat() if alert.last_checked else None,
        "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None,
        "trigger_count":  alert.trigger_count,
        "created_at": alert.created_at.isoformat() if alert.created_at else None,
        "last_result": json.loads(alert.last_result_json) if alert.last_result_json else None,
    }


# ── CRUD ────────────────────────────────────────────────────────────────────

def create_alert(
    db: Session,
    user_id: int,
    name: str,
    bbox: Dict[str, float],
    thresholds: Dict[str, Optional[float]],
    email: str,
    frequency: str,
    baseline: Dict[str, float],
) -> MonitoringAlert:
    alert = MonitoringAlert(
        user_id=user_id,
        name=name,
        bbox_west=bbox["west"],
        bbox_south=bbox["south"],
        bbox_east=bbox["east"],
        bbox_north=bbox["north"],
        thr_ndvi_drop=thresholds.get("ndvi_drop"),
        thr_ndbi_rise=thresholds.get("ndbi_rise"),
        thr_ndwi_change=thresholds.get("ndwi_change"),
        thr_area_ha=thresholds.get("area_ha"),
        email=email,
        frequency=frequency,
        baseline_json=json.dumps(baseline),
        status="active",
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    logger.info("Created monitoring alert id=%s '%s' for user %s", alert.id, name, user_id)
    return alert


def get_alerts_for_user(db: Session, user_id: int) -> List[MonitoringAlert]:
    return (
        db.query(MonitoringAlert)
        .filter(MonitoringAlert.user_id == user_id)
        .order_by(MonitoringAlert.created_at.desc())
        .all()
    )


def get_alert(db: Session, alert_id: int, user_id: int) -> Optional[MonitoringAlert]:
    return (
        db.query(MonitoringAlert)
        .filter(MonitoringAlert.id == alert_id, MonitoringAlert.user_id == user_id)
        .first()
    )


def get_all_active_alerts(db: Session) -> List[MonitoringAlert]:
    return db.query(MonitoringAlert).filter(MonitoringAlert.status == "active").all()


def update_alert_status(db: Session, alert: MonitoringAlert, status: str) -> None:
    alert.status = status
    db.commit()


def delete_alert(db: Session, alert: MonitoringAlert) -> None:
    db.delete(alert)
    db.commit()


# ── Threshold evaluator ─────────────────────────────────────────────────────

def evaluate_thresholds(
    alert: MonitoringAlert,
    current: Dict[str, float],
    area_ha: float,
    baseline_override: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Compare current spectral values against the alert's baseline + thresholds.
    Returns a list of breached threshold dicts (empty = no breach).
    """
    baseline = baseline_override or _baseline(alert)
    breached: List[Dict] = []

    ndvi_base = baseline.get("ndvi", 0.0)
    ndbi_base = baseline.get("ndbi", 0.0)
    ndwi_base = baseline.get("ndwi", 0.0)

    ndvi_now = current.get("ndvi", ndvi_base)
    ndbi_now = current.get("ndbi", ndbi_base)
    ndwi_now = current.get("ndwi", ndwi_base)

    if alert.thr_ndvi_drop is not None:
        delta = ndvi_base - ndvi_now          # positive = drop
        if delta >= alert.thr_ndvi_drop:
            breached.append({
                "metric":    "NDVI (vegetation health)",
                "baseline":  ndvi_base,
                "current":   ndvi_now,
                "delta":     ndvi_now - ndvi_base,
                "threshold": f"drop ≥ {alert.thr_ndvi_drop}",
            })

    if alert.thr_ndbi_rise is not None:
        delta = ndbi_now - ndbi_base          # positive = rise
        if delta >= alert.thr_ndbi_rise:
            breached.append({
                "metric":    "NDBI (built-up intensity)",
                "baseline":  ndbi_base,
                "current":   ndbi_now,
                "delta":     ndbi_now - ndbi_base,
                "threshold": f"rise ≥ {alert.thr_ndbi_rise}",
            })

    if alert.thr_ndwi_change is not None:
        delta = abs(ndwi_now - ndwi_base)
        if delta >= alert.thr_ndwi_change:
            breached.append({
                "metric":    "NDWI (water presence)",
                "baseline":  ndwi_base,
                "current":   ndwi_now,
                "delta":     ndwi_now - ndwi_base,
                "threshold": f"|Δ| ≥ {alert.thr_ndwi_change}",
            })

    if alert.thr_area_ha is not None and area_ha >= alert.thr_area_ha:
        breached.append({
            "metric":    "Changed area",
            "baseline":  0.0,
            "current":   area_ha,
            "delta":     area_ha,
            "threshold": f"≥ {alert.thr_area_ha} ha",
        })

    return breached
