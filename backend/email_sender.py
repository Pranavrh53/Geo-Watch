"""
Geo-Watch alert email sender.
Uses smtplib with Gmail App Password — no paid email service required.
"""
import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)


def send_alert_email(
    to_email: str,
    alert_name: str,
    bbox: dict,
    baseline: dict,
    current: dict,
    breached: list,
    area_ha: float,
    is_test: bool = False,
) -> None:
    """
    Send an HTML change-alert email.

    breached: list of dicts with keys metric, baseline, current, delta, threshold
    """
    # Read credentials fresh each call (so .env changes take effect after restart)
    ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
    ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
    APP_URL = os.getenv("APP_URL", "http://localhost:8000")

    if not ALERT_EMAIL_FROM or not ALERT_EMAIL_PASSWORD:
        raise ValueError(
            "Email credentials not configured. "
            "Set ALERT_EMAIL_FROM and ALERT_EMAIL_PASSWORD in your .env file."
        )

    subject = (
        f"🧪 [TEST] Geo-Watch Alert: {alert_name}"
        if is_test
        else f"🚨 Geo-Watch Alert: {alert_name} — Change Detected"
    )

    # ── Breached thresholds table rows ──
    breached_rows = ""
    for b in breached:
        delta_str = f"{b['delta']:+.4f}"
        breached_rows += f"""
        <tr>
            <td style="padding:8px 12px;border-bottom:1px solid #2a2a2a;color:#e0e0e0;">{b['metric']}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #2a2a2a;color:#aaa;">{b['baseline']:.4f}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #2a2a2a;color:#ff5555;font-weight:bold;">{b['current']:.4f}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #2a2a2a;color:#ff5555;">{delta_str}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #2a2a2a;color:#aaa;font-size:0.88em;">{b['threshold']}</td>
        </tr>"""

    test_banner = (
        '<div style="background:#5555FF;color:#fff;text-align:center;padding:10px;'
        'font-family:monospace;font-size:0.85em;font-weight:bold;margin-bottom:0;">'
        "🧪 TEST EMAIL — Sent manually via Run Check Now</div>"
        if is_test
        else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#0d0d0d;font-family:'Courier New',Courier,monospace;">
  {test_banner}
  <div style="max-width:660px;margin:0 auto;padding:28px 20px;">

    <!-- Header -->
    <div style="border:2px solid #ff5555;background:#111;padding:22px 24px;margin-bottom:20px;">
      <div style="font-size:0.7em;color:#ff5555;letter-spacing:2px;margin-bottom:4px;">GEO-WATCH SATELLITE MONITORING</div>
      <div style="font-size:1.35em;font-weight:bold;color:#fff;">🚨 CHANGE ALERT</div>
      <div style="font-size:1.05em;color:#b2e600;margin-top:6px;">{alert_name}</div>
      <div style="font-size:0.75em;color:#888;margin-top:8px;">
        Triggered: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
      </div>
    </div>

    <!-- Region -->
    <div style="background:#1a1a1a;border:1px solid #2a2a2a;padding:14px 18px;margin-bottom:16px;">
      <div style="color:#b2e600;font-size:0.7em;font-weight:bold;letter-spacing:1px;margin-bottom:8px;">📍 MONITORED REGION</div>
      <div style="font-size:0.82em;color:#aaa;">
        N {bbox.get('north',0):.5f}° &nbsp;·&nbsp; S {bbox.get('south',0):.5f}° &nbsp;·&nbsp;
        E {bbox.get('east',0):.5f}° &nbsp;·&nbsp; W {bbox.get('west',0):.5f}°
      </div>
    </div>

    <!-- Breached thresholds -->
    <div style="background:#1a1a1a;border:2px solid #ff5555;padding:16px 18px;margin-bottom:16px;">
      <div style="color:#ff5555;font-size:0.7em;font-weight:bold;letter-spacing:1px;margin-bottom:12px;">⚠️ THRESHOLDS BREACHED</div>
      <table style="width:100%;border-collapse:collapse;font-size:0.8em;">
        <thead>
          <tr style="background:#222;">
            <th style="padding:8px 12px;text-align:left;color:#888;font-weight:normal;">Metric</th>
            <th style="padding:8px 12px;text-align:left;color:#888;font-weight:normal;">Baseline</th>
            <th style="padding:8px 12px;text-align:left;color:#888;font-weight:normal;">Current</th>
            <th style="padding:8px 12px;text-align:left;color:#888;font-weight:normal;">Δ Change</th>
            <th style="padding:8px 12px;text-align:left;color:#888;font-weight:normal;">Trigger</th>
          </tr>
        </thead>
        <tbody>{breached_rows or '<tr><td colspan="5" style="padding:10px;color:#888;text-align:center;">Test check — no thresholds breached</td></tr>'}</tbody>
      </table>
    </div>

    <!-- Spectral values -->
    <div style="background:#1a1a1a;border:1px solid #2a2a2a;padding:16px 18px;margin-bottom:16px;">
      <div style="color:#b2e600;font-size:0.7em;font-weight:bold;letter-spacing:1px;margin-bottom:14px;">📊 CURRENT SPECTRAL VALUES</div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <div style="flex:1;min-width:130px;background:#111;border:1px solid #2a2a2a;padding:12px;text-align:center;">
          <div style="font-size:1.15em;font-weight:bold;color:#4caf50;">{current.get('ndvi',0):.4f}</div>
          <div style="font-size:0.68em;color:#888;margin-top:4px;">NDVI · Vegetation</div>
          <div style="font-size:0.65em;color:#555;">baseline {baseline.get('ndvi',0):.4f}</div>
        </div>
        <div style="flex:1;min-width:130px;background:#111;border:1px solid #2a2a2a;padding:12px;text-align:center;">
          <div style="font-size:1.15em;font-weight:bold;color:#f5a623;">{current.get('ndbi',0):.4f}</div>
          <div style="font-size:0.68em;color:#888;margin-top:4px;">NDBI · Built-up</div>
          <div style="font-size:0.65em;color:#555;">baseline {baseline.get('ndbi',0):.4f}</div>
        </div>
        <div style="flex:1;min-width:130px;background:#111;border:1px solid #2a2a2a;padding:12px;text-align:center;">
          <div style="font-size:1.15em;font-weight:bold;color:#2196f3;">{current.get('ndwi',0):.4f}</div>
          <div style="font-size:0.68em;color:#888;margin-top:4px;">NDWI · Water</div>
          <div style="font-size:0.65em;color:#555;">baseline {baseline.get('ndwi',0):.4f}</div>
        </div>
        <div style="flex:1;min-width:130px;background:#111;border:1px solid #2a2a2a;padding:12px;text-align:center;">
          <div style="font-size:1.15em;font-weight:bold;color:#ff5555;">{area_ha:.1f} ha</div>
          <div style="font-size:0.68em;color:#888;margin-top:4px;">Changed Area</div>
          <div style="font-size:0.65em;color:#555;">estimated</div>
        </div>
      </div>
    </div>

    <!-- CTA button -->
    <div style="text-align:center;margin:24px 0;">
      <a href="{APP_URL}/frontend/compare.html"
         style="display:inline-block;padding:13px 32px;background:#b2e600;color:#000;
                font-weight:bold;font-family:'Courier New',monospace;font-size:0.85em;
                text-decoration:none;letter-spacing:1px;">
        → VIEW ON GEO-WATCH MAP
      </a>
    </div>

    <!-- Footer -->
    <div style="font-size:0.65em;color:#444;border-top:1px solid #1f1f1f;padding-top:14px;line-height:1.8;">
      This alert was automatically generated by Geo-Watch Satellite Change Detection System.<br>
      Data source: Sentinel-2 L2A · Copernicus Dataspace · 10 m resolution<br>
      Spectral indices: NDVI (B08−B04)/(B08+B04) · NDBI (B11−B08)/(B11+B08) · NDWI (B03−B08)/(B03+B08)<br>
      To pause or delete this alert, visit your Geo-Watch Alerts dashboard.
    </div>
  </div>
</body>
</html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Geo-Watch Alerts <{ALERT_EMAIL_FROM}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD)
            server.sendmail(ALERT_EMAIL_FROM, to_email, msg.as_string())
        logger.info("Alert email sent to %s for alert '%s'", to_email, alert_name)
    except smtplib.SMTPAuthenticationError:
        raise ValueError(
            "Gmail authentication failed. Make sure ALERT_EMAIL_PASSWORD is a Gmail App Password "
            "(not your account password). Generate one at https://myaccount.google.com/apppasswords"
        )
    except Exception as exc:
        logger.error("Failed to send alert email to %s: %s", to_email, exc)
        raise
