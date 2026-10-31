#!/usr/bin/env bash
HSTATS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HSTATS_DIR/h-manifest.conf"
log_name="${CUSTOM_LOG_BASENAME}.log"
[[ -f $log_name ]] || exit 0
diffTime=$(( $(date +%s) - $(stat -c %Y "$log_name") ))
(( diffTime > 180 )) && exit 0
tail_text=$(tail -n 300 "$log_name" | sed -e "s/\x1b\[[0-9;]*m//g")
line=$(echo "$tail_text" | grep "hashrate" | tail -n 1)
[[ -z $line ]] && exit 0
total=$(echo "$line" | grep -oP 'hashrate \K[0-9]+(\.[0-9]+)?')
g0=$(echo "$line" | grep -oP 'GPU0 \K[0-9]+(\.[0-9]+)?')
g1=$(echo "$line" | grep -oP 'GPU1 \K[0-9]+(\.[0-9]+)?')
asr=$(echo "$line" | grep -oP 'A/S/R \K[0-9]+/[0-9]+/[0-9]+')
ac=${asr%%/*}; rest=${asr#*/}; st=${rest%%/*}; rj=${rest##*/}
hh=$(echo "$line" | sed -nE 's/.*up ([0-9]+)h.*/\1/p')
mm=$(echo "$line" | sed -nE 's/.*up ([0-9]+)m.*/\1/p'; echo "$line" | sed -nE 's/.*up [0-9]+h ([0-9]+)m.*/\1/p' | head -1)
[[ -z $hh ]] && hh=0
[[ -z $mm ]] && mm=$(echo "$line" | grep -oP 'up \K[0-9]+(?=m)')
[[ -z $mm ]] && mm=0
up=$(( hh * 3600 + mm * 60 ))
cfg="${CUSTOM_CONFIG_FILENAME:-/tmp/prlminer.conf}"
wallet=$(grep -hoP 'prl1[a-z0-9]+' "$cfg" 2>/dev/null | head -1)
hs=$(jq -cn --arg a "${g0:-0}" --arg b "${g1:-0}" '[((($a|tonumber)*1e12)|floor), ((($b|tonumber)*1e12)|floor)]')
total_h=$(jq -cn --arg t "${total:-0}" '$t|tonumber*1e12|floor')
temps=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | jq -cs '.')
fans=$(nvidia-smi --query-gpu=fan.speed --format=csv,noheader,nounits | jq -cs '.')
url=$(echo "$tail_text" | grep -oP 'Connected to \K\S+' | tail -n 1)
worker=$(grep -hoP 'CUSTOM_TEMPLATE="[^"]*\.\K[^"]+' "$cfg" 2>/dev/null | head -1)
[[ -z $worker ]] && worker="${WORKER_NAME:-Farm-G1}"
jq -nc --argjson hs "$hs" --argjson temp "$temps" --argjson fan "$fans" \
  --argjson khs "$total_h" --arg ac "$ac" --arg rj "$rj" --arg st "${st:-0}" \
  --arg uptime "$up" --arg url "$url" --arg user "$wallet" \
  --arg worker "$worker" --arg ver "0.1.0" \
  '{hs_units:"hs", hs:$hs, khs:$khs, temp:$temp, fan:$fan, ar:[$ac|tonumber,$rj|tonumber], as:$st,
    uptime:$uptime, ver:$ver, dual:false, pools:$url,
    workers:[{url:$url, alive:true, user:$user, worker:$worker}], rejected_share:0, invalid_share:0,
    accepted_share:($ac|tonumber), stale_share:($st|tonumber)}'
