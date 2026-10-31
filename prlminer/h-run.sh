#!/usr/bin/env bash
# HiveOS custom-miner launcher for prlminer.
#
# Flight sheet mapping:
#   Pool url                -> $CUSTOM_URL
#   Wallet and worker tmpl. -> $CUSTOM_TEMPLATE   (prl1... or krxWALLET.worker)
#   Extra config arguments  -> $CUSTOM_USER_CONFIG (passed through verbatim)
#   Rig name                -> $WORKER_NAME
cd "$(dirname "$0")" || exit 1
. ./h-config.sh

# HiveOS restarts us on any flight-sheet change: make sure a previous instance is gone first.
# `-x` matches the process name exactly, so it can never hit the invoking shell (whose command
# line contains the path ".../prlminer/...").
pkill -x prlminer >/dev/null 2>&1 && sleep 2

# HiveOS's "Wallet and worker template" may already carry a worker ("wallet.worker"); only add the
# rig name when it does not, so the pool login is never doubled into "wallet.worker.worker".
worker_args=()
case "$CUSTOM_TEMPLATE" in
  *.*) ;;
  *) worker_args=(-w "${WORKER_NAME:-rig}") ;;
esac

# No tty under HiveOS -> plain log; merge stderr so HiveOS captures everything.
exec ./prlminer \
  -o "${CUSTOM_URL:-stratum+ssl://prl-eu.kryptex.network:8048}" \
  -u "${CUSTOM_TEMPLATE}" \
  "${worker_args[@]}" \
  --no-tui --stats 30 --log-file /var/log/miner/prlminer/prlminer.log \
  ${CUSTOM_USER_CONFIG:-} 2>&1
