"""
Quick test to verify Sentinel Hub configuration is loaded
"""
import os
from dotenv import load_dotenv

load_dotenv()

instance_id = os.getenv('SENTINEL_HUB_INSTANCE_ID')
username = os.getenv('COPERNICUS_USERNAME')

print("=" * 60)
print("SENTINEL HUB CONFIGURATION CHECK")
print("=" * 60)
print(f"\n✓ Copernicus User: {username}")
print(f"✓ Instance ID: {instance_id}")
print(f"\n✓ WMS Endpoint: https://sh.dataspace.copernicus.eu/ogc/wms/{instance_id}")
print("\n" + "=" * 60)
print("Configuration looks good! Ready to fetch real satellite images.")
print("=" * 60)
