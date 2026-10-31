#!/usr/bin/env bash
# HiveOS sources this before h-run.sh: it loads the flight-sheet settings into the environment.
# HiveOS exports CUSTOM_CONFIG_FILENAME (the flight sheet rendered as a shell file); sourcing it
# provides CUSTOM_URL, CUSTOM_TEMPLATE, CUSTOM_USER_CONFIG and friends.
[[ -e "$CUSTOM_CONFIG_FILENAME" ]] && . "$CUSTOM_CONFIG_FILENAME"
