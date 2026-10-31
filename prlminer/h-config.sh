#!/usr/bin/env bash
# Renders $CUSTOM_CONFIG_FILENAME. Works in agent CONFIG phase (env exported),
# in RUN phase (re-sources /hive-config/wallet.conf), and standalone.
[[ -z "$CUSTOM_CONFIG_FILENAME" ]] && return 0
if [[ -z "$CUSTOM_URL" && -z "$CUSTOM_TEMPLATE" && -r /hive-config/wallet.conf ]]; then
  . /hive-config/wallet.conf 2>/dev/null || true
fi
[[ -z "$CUSTOM_URL" && -z "$CUSTOM_TEMPLATE" ]] && return 0   # nothing to render: keep existing
[[ "$CUSTOM_TEMPLATE" == *.* ]] && CUSTOM_TEMPLATE="${CUSTOM_TEMPLATE/./\/}"
{
  echo "CUSTOM_URL=\"$CUSTOM_URL\""
  echo "CUSTOM_TEMPLATE=\"$CUSTOM_TEMPLATE\""
  echo "CUSTOM_PASS=\"${CUSTOM_PASS:-x}\""
  echo "CUSTOM_USER_CONFIG=\"$CUSTOM_USER_CONFIG\""
} > "$CUSTOM_CONFIG_FILENAME"
return 0
