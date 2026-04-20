"""
Geo-Watch background scheduler — checks monitoring alerts on a schedule.
Uses APScheduler (BackgroundScheduler) embedded in the FastAPI process.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None


def run_alert_check(alert_id: int, db_factory, detector_factory) -> dict:
    """
    Core check logic for one alert. Called both by scheduler and by
    the "Run Check Now" API endpoint so demo works instantly.

    Returns a result dict with keys: breached, current, area_ha, email_sent, error
    """
    from backend.alerts import (
        get_all_active_alerts, evaluate_thresholds,
        update_alert_status, alert_to_dict,
    )
    from backend.email_sender import send_alert_email

    db: Session = db_factory()
    result = {"breached": [], "current": {}, "area_ha": 0.0, "email_sent": False, "error": None}

    try:
        # Load the specific alert
        from backend.alerts import MonitoringAlert
        alert = db.query(MonitoringAlert).filter(MonitoringAlert.id == alert_id).first()
        if not alert:
            result["error"] = f"Alert {alert_id} not found"
            return result
        if alert.status == "paused":
            result["error"] = "Alert is paused"
            return result

        bbox = {
            "west": alert.bbox_west, "south": alert.bbox_south,
            "east": alert.bbox_east, "north": alert.bbox_north,
        }

        # Fetch current spectral data using the unified detector
        detector = detector_factory()
        import numpy as np
        from datetime import datetime as dt

        def _yearly_means(yearly_obj):
            mask = yearly_obj.valid_mask
            ndvi_vals = yearly_obj.ndvi[mask] if np.any(mask) else yearly_obj.ndvi.ravel()
            ndbi_vals = yearly_obj.ndbi[mask] if np.any(mask) else yearly_obj.ndbi.ravel()
            ndwi_vals = yearly_obj.ndwi[mask] if np.any(mask) else yearly_obj.ndwi.ravel()
            return {
                "ndvi": round(float(np.nanmean(ndvi_vals)), 4),
                "ndbi": round(float(np.nanmean(ndbi_vals)), 4),
                "ndwi": round(float(np.nanmean(ndwi_vals)), 4),
            }

        def _valid_stats(yearly_obj):
            valid_count = int(np.sum(yearly_obj.valid_mask))
            total_count = int(yearly_obj.valid_mask.size)
            valid_fraction = float(valid_count / max(1, total_count))
            return {
                "valid_count": valid_count,
                "total_count": total_count,
                "valid_fraction": valid_fraction,
                "cloud_percent": float(yearly_obj.cloud_percent),
            }

        now_utc = dt.utcnow()
        frequency = (alert.frequency or "weekly").lower()
        window_days_by_frequency = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
        }
        window_days = window_days_by_frequency.get(frequency, 7)

        candidate_windows = {
            "daily": [1, 3, 7],
            "weekly": [7, 14],
            "monthly": [30, 45],
        }.get(frequency, [window_days])

        # Rough area estimate from NDBI rise pixels
        import math
        lat_m = (bbox["north"] - bbox["south"]) * 111_000
        avg_lat = math.radians((bbox["north"] + bbox["south"]) / 2)
        lon_m = (bbox["east"] - bbox["west"]) * 111_000 * math.cos(avg_lat)
        total_m2 = lat_m * lon_m
        pixel_area_m2 = total_m2 / max(1, detector.model_size ** 2)

        min_valid_fraction = 0.15
        max_cloud_percent = 85.0
        selected = None
        last_quality = None
        last_reason = ""

        for candidate_days in candidate_windows:
            current_start = now_utc - timedelta(days=candidate_days)
            baseline_end = current_start
            baseline_start = baseline_end - timedelta(days=candidate_days)

            try:
                current_data = detector._fetch_window_data(
                    bbox=bbox,
                    start_dt=current_start,
                    end_dt=now_utc,
                    size=detector.model_size,
                )
                baseline_data = detector._fetch_window_data(
                    bbox=bbox,
                    start_dt=baseline_start,
                    end_dt=baseline_end,
                    size=detector.model_size,
                )
            except Exception as fetch_exc:
                last_reason = f"Window fetch failed for {candidate_days}d: {fetch_exc}"
                logger.warning("Alert %s window fetch failed (%sd): %s", alert_id, candidate_days, fetch_exc)
                continue

            if current_data.is_synthetic or baseline_data.is_synthetic:
                last_reason = "Synthetic/demo data returned"
                continue

            current_stats = _valid_stats(current_data)
            baseline_stats = _valid_stats(baseline_data)
            last_quality = {
                "current": current_stats,
                "baseline": baseline_stats,
                "candidate_days": candidate_days,
            }

            quality_ok = (
                current_stats["valid_fraction"] >= min_valid_fraction
                and baseline_stats["valid_fraction"] >= min_valid_fraction
                and current_stats["cloud_percent"] <= max_cloud_percent
                and baseline_stats["cloud_percent"] <= max_cloud_percent
            )
            if not quality_ok:
                last_reason = "Data quality below threshold"
                continue

            selected = {
                "days": candidate_days,
                "current_start": current_start,
                "baseline_start": baseline_start,
                "baseline_end": baseline_end,
                "current_data": current_data,
                "baseline_data": baseline_data,
                "current_stats": current_stats,
                "baseline_stats": baseline_stats,
            }
            break

        if not selected:
            result["error"] = (
                "Data quality too low for reliable change check; "
                "skipping alert trigger"
            )
            alert.last_checked = datetime.utcnow()
            import json as _json
            alert.last_result_json = _json.dumps({
                "checked_at": datetime.utcnow().isoformat(),
                "frequency": frequency,
                "data_quality": "insufficient",
                "reason": last_reason,
                "quality": {
                    "last_attempt": last_quality,
                    "min_valid_fraction": min_valid_fraction,
                    "max_cloud_percent": max_cloud_percent,
                },
                "window_candidates": candidate_windows,
            })
            if alert.status == "triggered":
                alert.status = "active"
            db.commit()
            return result

        window_days = selected["days"]
        current_start = selected["current_start"]
        baseline_start = selected["baseline_start"]
        baseline_end = selected["baseline_end"]
        yearly_data = selected["current_data"]
        baseline_data = selected["baseline_data"]
        current_stats = selected["current_stats"]
        baseline_stats = selected["baseline_stats"]

        valid = yearly_data.valid_mask
        current = _yearly_means(yearly_data)
        result["current"] = current

        import json

        # Refresh persisted baseline from the exact baseline year used in comparisons.
        baseline = _yearly_means(baseline_data)
        baseline["window"] = {
            "from": baseline_start.isoformat() + "Z",
            "to": baseline_end.isoformat() + "Z",
            "days": window_days,
            "frequency": frequency,
        }
        alert.baseline_json = json.dumps(baseline)

        # Compute changed area from true year-over-year per-pixel deltas.
        both_valid = valid & baseline_data.valid_mask
        ndvi_delta = np.abs(yearly_data.ndvi - baseline_data.ndvi)
        ndbi_delta = np.abs(yearly_data.ndbi - baseline_data.ndbi)
        ndwi_delta = np.abs(yearly_data.ndwi - baseline_data.ndwi)

        # Conservative per-pixel delta gates reduce noise-driven false positives.
        changed_mask = both_valid & (
            (ndvi_delta >= 0.08) |
            (ndbi_delta >= 0.05) |
            (ndwi_delta >= 0.08)
        )
        changed_pixels = int(np.sum(changed_mask))
        area_ha = round(changed_pixels * pixel_area_m2 / 10_000, 2)
        result["area_ha"] = area_ha

        # Evaluate thresholds
        breached = evaluate_thresholds(alert, current, area_ha, baseline_override=baseline)
        result["breached"] = breached

        # Update DB
        alert.last_checked = datetime.utcnow()
        import json as _json
        alert.last_result_json = _json.dumps({
            "checked_at": datetime.utcnow().isoformat(),
            "frequency": frequency,
            "current": current,
            "baseline": baseline,
            "area_ha": area_ha,
            "breached_count": len(breached),
            "data_quality": "real",
            "quality": {
                "current": current_stats,
                "baseline": baseline_stats,
            },
            "windows": {
                "current": {
                    "from": current_start.isoformat() + "Z",
                    "to": now_utc.isoformat() + "Z",
                    "days": window_days,
                },
                "baseline": {
                    "from": baseline_start.isoformat() + "Z",
                    "to": baseline_end.isoformat() + "Z",
                    "days": window_days,
                },
            },
        })

        if breached:
            alert.last_triggered = datetime.utcnow()
            alert.trigger_count = (alert.trigger_count or 0) + 1
            alert.status = "triggered"
            db.commit()

            # Send email
            try:
                send_alert_email(
                    to_email=alert.email,
                    alert_name=alert.name,
                    bbox=bbox,
                    baseline=baseline,
                    current=current,
                    breached=breached,
                    area_ha=area_ha,
                )
                result["email_sent"] = True
                logger.info("Alert %s triggered — email sent to %s", alert_id, alert.email)
            except Exception as email_exc:
                result["error"] = f"Thresholds breached but email failed: {email_exc}"
                logger.error("Email send failed for alert %s: %s", alert_id, email_exc)
        else:
            # Reset to active if it was previously triggered
            if alert.status == "triggered":
                alert.status = "active"
            db.commit()
            logger.info("Alert %s checked — no breach (NDVI=%.4f NDBI=%.4f)", alert_id, current["ndvi"], current["ndbi"])

    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Alert check failed for alert %s: %s", alert_id, exc, exc_info=True)
    finally:
        db.close()

    return result


def _scheduled_check_all(db_factory, detector_factory) -> None:
    """Run checks for all due alerts — respects each alert's frequency setting."""
    from backend.alerts import MonitoringAlert

    FREQ_HOURS = {"daily": 23, "weekly": 167, "monthly": 719}  # slightly under to avoid drift

    db = db_factory()
    try:
        alerts = db.query(MonitoringAlert).filter(
            MonitoringAlert.status.in_(["active", "triggered"])
        ).all()

        now = datetime.utcnow()
        due_ids = []
        for a in alerts:
            min_gap = timedelta(hours=FREQ_HOURS.get(a.frequency, 23))
            if a.last_checked is None or (now - a.last_checked) >= min_gap:
                due_ids.append(a.id)
    finally:
        db.close()

    logger.info("Scheduler: %d active alerts, %d due for check", len(alerts) if 'alerts' in dir() else 0, len(due_ids))
    for aid in due_ids:
        run_alert_check(aid, db_factory, detector_factory)


def start_scheduler(db_factory, detector_factory) -> None:
    """Start the background scheduler. Call once on app startup."""
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler(timezone="UTC")

    # Run every 24 hours — for demo purposes you can lower this
    _scheduler.add_job(
        _scheduled_check_all,
        trigger=IntervalTrigger(hours=24),
        args=[db_factory, detector_factory],
        id="daily_alert_check",
        replace_existing=True,
        next_run_time=None,   # don't run immediately on startup
    )

    _scheduler.start()
    logger.info("Alert scheduler started (24-hour interval)")


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Alert scheduler stopped")
