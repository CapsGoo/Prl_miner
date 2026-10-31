#!/usr/bin/env python3
# Pearl (PRL) live profit calculator — rig-level, fee-aware
import json, urllib.request, sys

NET_NETHASH_TH = 78_752_630_000 / 1e3   # 78.75 EH/s -> TH/s (from pool/whattomine; auto-refresh below)
BLOCK_TIME_S = 179.0
BLOCK_REWARD_PRL = 2207.0

def get_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=pearl-2&vs_currencies=usd"
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)["pearl-2"]["usd"]

def get_kryptex_stats():
    try:
        url = "https://pool.kryptex.com/api/v2/stat/coins"
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.load(r)
        for c in d if isinstance(d, list) else d.get("coins", []):
            if str(c.get("symbol", c.get("coin", ""))).upper() == "PRL":
                return c
    except Exception:
        pass
    return None

def rig_day_usd(rig_th, watts, price, net_th, reward, pool_fee, dev_fee, elec, kwh=0.08):
    share = rig_th / net_th
    prl_day = share * reward * 86400 / BLOCK_TIME_S
    gross = prl_day * price
    net = gross * (1 - pool_fee) * (1 - dev_fee)
    power_cost = watts / 1000.0 * 24 * kwh
    return prl_day, gross, net, net - power_cost

def main():
    price = get_price()
    kx = get_kryptex_stats()
    net_th = NET_NETHASH_TH
    reward = BLOCK_REWARD_PRL
    if kx:
        for k in ("nethash", "network_hashrate"):
            if kx.get(k):
                v = float(kx[k])
                net_th = v / 1e12 if v > 1e9 else v
        if kx.get("block_reward"):
            reward = float(kx["block_reward"])
    print(f"PRL price ${price}  network {net_th:,.0f} TH/s  reward {reward}/block")
    scen = [
        ("prlminer @ Kryptex (3% + 0.5% fee)", 151.7, 520, 0.03, 0.005),
        ("KRig @ Kryptex (3% + 0 fee)",        151.7, 520, 0.03, 0.0),
        ("GhostMiner @ Kryptex (3% + 0)",      165.0, 480, 0.03, 0.0),
        ("GhostMiner @ Kryptex (+30% eff)",    197.0, 480, 0.03, 0.0),
    ]
    kwh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
    print(f"electricity {kwh} $/kWh")
    for name, th, w, pf, df in scen:
        prl, gross, net, prof = rig_day_usd(th, w, price, net_th, reward, pf, df, prof_ := None, kwh)
        print(f"{name:38s} {th:6.1f} TH  {prl:8.2f} PRL/d  gross ${gross:5.2f}  net ${net:5.2f}  PROFIT ${prof:5.2f}/d  ({prl*30:.0f} PRL/mo)")

if __name__ == "__main__":
    main()
