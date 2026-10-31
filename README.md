# prlminer — HiveOS custom miner package (PRL)

Source-of-truth package files for HiveOS custom-miner install.

- `prlminer/` — package contents (binary + h-manifest.conf, h-config.sh, h-run.sh, h-stats.sh)
- `pack.sh` — builds `dist/prlminer-<VER>-hiveos.tar.gz`
- HiveOS install: Flight Sheet → Miner → Custom → Installation URL → release asset URL (`prlminer-x.y.z-hiveos.tar.gz`)

## Update flow
1. bump `CUSTOM_VERSION` in `prlminer/h-manifest.conf`, replace binary/scripts
2. `bash pack.sh`
3. upload `dist/*.tar.gz` as a GitHub Release asset
4. point the Installation URL to `https://github.com/CapsGoo/Prl_miner/releases/download/<tag>/prlminer-<VER>-hiveos.tar.gz`

## Notes
- `h-stats.sh` feeds HiveOS panel stats (hashrate/temp/fan/shares) from `/var/log/miner/prlminer/prlminer.log`
- `h-run.sh` takes pool/wallet from the flight sheet (`CUSTOM_URL`, `CUSTOM_TEMPLATE`, `CUSTOM_USER_CONFIG`)
- upstream miner: github.com/vv7r/prlminer (ISC-based parts, 0.5% dev fee)
