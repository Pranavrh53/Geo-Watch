# Network Troubleshooting Guide for Real Satellite Images

## Problem
You're seeing demo images because the system cannot connect to Copernicus Data Space servers due to network timeouts.

## Solutions

### Option 1: Check if You're on a Restricted Network (Most Common)

**Are you connected to:**
- College/University WiFi (like RVEI network)?
- Corporate network?
- Public WiFi with restrictions?

**If YES**, the network likely blocks external API connections.

**Solution**: Try switching to:
- Mobile hotspot
- Home WiFi
- Different network without restrictions

---

### Option 2: Configure Proxy Settings (If Required by Network)

If your network requires a proxy:

1. **Find Proxy Settings**:
   - Windows Settings → Network & Internet → Proxy
   - Note the proxy address (e.g., `http://proxy.company.com:8080`)

2. **Update .env file** - Add these lines:
```env
HTTP_PROXY=http://your-proxy:8080
HTTPS_PROXY=http://your-proxy:8080
```

3. **If proxy requires authentication**:
```env
HTTP_PROXY=http://username:password@your-proxy:8080
HTTPS_PROXY=http://username:password@your-proxy:8080
```

4. **Restart backend server**

---

### Option 3: Check Windows Firewall

1. Open Windows Security → Firewall & network protection
2. Click "Allow an app through firewall"
3. Click "Change settings" → "Allow another app"
4. Find and add: `python.exe` from your venv folder
5. Check both "Private" and "Public" boxes

---

### Option 4: Disable Antivirus Temporarily

Some antivirus software blocks Python's network requests:
1. Temporarily disable your antivirus
2. Try fetching images again
3. If it works, add Python to antivirus whitelist

---

### Option 5: Check Internet Connection

Test if you can reach Copernicus website:
1. Open browser
2. Go to: https://dataspace.copernicus.eu/
3. If it doesn't load, there's a general internet issue

---

### Option 6: Use Mobile Hotspot (Quick Test)

**Fastest way to test if it's your network:**
1. Create mobile hotspot from your phone
2. Connect your PC to the hotspot
3. Restart backend: `python backend\main.py`
4. Try fetching images again

If it works on mobile hotspot = Your regular network is blocking the connection

---

### Option 7: Contact Network Administrator

If on college/corporate network:
- Ask IT to whitelist: `identity.dataspace.copernicus.eu`
- Ask IT to whitelist: `sh.dataspace.copernicus.eu`
- Port required: 443 (HTTPS)

---

## Testing After Each Solution

After trying any solution:

1. **Clear cache**:
```powershell
Remove-Item data\tile_cache\*.png -Force
```

2. **Restart backend**:
```powershell
python backend\main.py
```

3. **Test connection**:
```powershell
python test_credentials.py
```
Should show: ✅ SUCCESS (not timeout error)

4. **Try fetching images** in a NEW region (not where you tested before)

---

## Current Workaround: Use Demo Mode

While demo images are synthetic, your system is still fully functional:
- User authentication ✅
- Interactive map ✅
- Region drawing ✅
- Image comparison ✅
- History tracking ✅
- All features work except real satellite data

Demo mode is perfect for:
- Testing the application
- Demonstrating the interface
- Development and debugging

---

## Most Likely Issue

Based on your college email (rvei.edu.in), you're probably on a **college network that blocks external API calls**. 

**Quick test**: Try connecting via mobile hotspot. If it works, you know it's the college network.

**Permanent solution**: Either use mobile hotspot when you need real images, or ask college IT to whitelist the Copernicus domains.
