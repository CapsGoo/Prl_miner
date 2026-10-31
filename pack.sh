#!/usr/bin/env bash
# Build HiveOS custom-miner archive from ./prlminer into ./dist/
set -e
cd "$(dirname "$0")"
VER=$(sed -n 's/^CUSTOM_VERSION=//p' prlminer/h-manifest.conf)
mkdir -p dist
rm -rf .build && mkdir -p .build
cp -R prlminer ".build/prlminer-$VER"
find .build -name '.DS_Store' -delete
tar --no-xattrs -czf "dist/prlminer-$VER-hiveos.tar.gz" -C .build "prlminer-$VER"
rm -rf .build
ls -la dist/
