#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
mkdir -p /var/log/miner/prlminer 2>/dev/null || true
: "${CUSTOM_CONFIG_FILENAME:=/tmp/prlminer.conf}"
[[ -f "$CUSTOM_CONFIG_FILENAME" ]] && . "$CUSTOM_CONFIG_FILENAME"
. ./h-config.sh
[[ -f "$CUSTOM_CONFIG_FILENAME" ]] && . "$CUSTOM_CONFIG_FILENAME"
[[ -f "$CUSTOM_CONFIG_FILENAME" ]] && . "$CUSTOM_CONFIG_FILENAME"
pkill -x prlminer >/dev/null 2>&1 && sleep 2
# split wallet/worker: Kryptex PRL needs -u wallet -w worker
if [[ "$CUSTOM_TEMPLATE" == *.* ]]; then
  WAL="${CUSTOM_TEMPLATE%%.*}"; WRK="${CUSTOM_TEMPLATE#*.}"
elif [[ "$CUSTOM_TEMPLATE" == */* ]]; then
  WAL="${CUSTOM_TEMPLATE%%/*}"; WRK="${CUSTOM_TEMPLATE#*/}"
else
  WAL="$CUSTOM_TEMPLATE"; WRK="${WORKER_NAME:-rig}"
fi
./prlminer \
  -o "${CUSTOM_URL:-stratum+tcp://prl-eu.kryptex.network:7048}" \
  -u "$WAL" -w "$WRK" -p "${CUSTOM_PASS:-x}" \
  --no-tui --stats 30 --log-file /var/log/miner/prlminer/prlminer.log \
  ${CUSTOM_USER_CONFIG:-} 2>&1
