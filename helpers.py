"""
Spike test for the Planview OData endpoint - SANDBOX ONLY.

Usage (token via environment variable, never hardcoded):
  Windows PowerShell:  $env:PV_TOKEN="your_token"; python planview_spike_sandbox.py smoke
  macOS/Linux/WSL:     PV_TOKEN=your_token python planview_spike_sandbox.py smoke

Modes:
  smoke  -> 1 user for 10s, just checks auth + query work (run this first)
  spike  -> baseline -> spike -> recovery
"""
import os, sys, csv, time, statistics, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ---- Config -----------------------------------------------------------------
SANDBOX_HOST = "hcsc-sb.pvcloud.com"
BASE = f"https://{SANDBOX_HOST}/odataservice/odataservice.svc/Automart_project_dim"

# Replace with real property names from $metadata if these don't match.
SELECT = "Workid,Worktype,WorkName,WorkDescription,WorkStatus"
URL = f"{BASE}?$select={SELECT}&$top=50&$format=json"

# (duration_seconds, concurrent_users) - kept deliberately modest
MODES = {
    "smoke": [(10, 1)],
    "spike": [(60, 2), (60, 20), (90, 2)],   # baseline -> spike -> recovery
}
PER_USER_DELAY = 1.0    # seconds between requests per user
TIMEOUT = 60

# ---- Safety guard: refuse anything that isn't the sandbox --------------------
if SANDBOX_HOST not in URL or "://hcsc.pvcloud.com" in URL:
    sys.exit("Refusing to run: URL is not the sandbox endpoint.")

token = os.environ.get("PV_TOKEN")
if not token:
    sys.exit("Set the PV_TOKEN environment variable first.")

HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {token}",   # change if the login doc uses a different header
}

# ---- Test run ---------------------------------------------------------------
lock = threading.Lock()
results = []          # (stage, timestamp, latency_s, status)
stop_all = threading.Event()

def one_request(stage):
    start = time.perf_counter()
    try:
        req = urllib.request.Request(URL, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception as e:
        status = f"ERR:{type(e).__name__}"
    latency = time.perf_counter() - start
    with lock:
        results.append((stage, datetime.now().isoformat(timespec="seconds"), latency, status))
    if status in (401, 403):
        stop_all.set()   # bad/expired token - stop immediately
    elif status in (429, 503):
        time.sleep(5)    # server throttling - back off

def run_stage(name, duration, users):
    end = time.time() + duration
    def worker():
        while time.time() < end and not stop_all.is_set():
            one_request(name)
            time.sleep(PER_USER_DELAY)
    with ThreadPoolExecutor(max_workers=users) as ex:
        for _ in range(users):
            ex.submit(worker)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode not in MODES:
        sys.exit(f"Unknown mode '{mode}'. Use: {', '.join(MODES)}")

    print(f"Target: {BASE}  (mode: {mode})")
    for i, (dur, users) in enumerate(MODES[mode]):
        if stop_all.is_set():
            break
        name = f"stage{i+1} ({users} users)"
        print(f"Running {name} for {dur}s...")
        run_stage(name, dur, users)

    if stop_all.is_set():
        print("\nStopped early: got 401/403. Check the token (expired or wrong header format).")

    print("\n--- Summary ---")
    for name in dict.fromkeys(r[0] for r in results):
        rows = [r for r in results if r[0] == name]
        lat = sorted(r[2] for r in rows)
        codes = {}
        for r in rows:
            codes[r[3]] = codes.get(r[3], 0) + 1
        ok = codes.get(200, 0)
        p95 = lat[max(int(len(lat) * 0.95) - 1, 0)]
        print(f"{name}: {len(rows)} reqs | success {ok/len(rows):.1%} | "
              f"p50 {statistics.median(lat):.2f}s | p95 {p95:.2f}s | max {lat[-1]:.2f}s | codes {codes}")

    out = f"spike_results_{mode}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "timestamp", "latency_s", "status"])
        w.writerows(results)
    print(f"\nRaw results saved to {out}")

if __name__ == "__main__":
    main()
