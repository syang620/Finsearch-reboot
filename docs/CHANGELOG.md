# Changelog

All notable repository updates can be tracked here.

## 2026-02-23
### Scripts and ingestion tooling
- Added `scripts/ingestion_cli.py` as a unified ingestion entrypoint.
- Added `scripts/_common.py` to centralize ticker/quarter parsing.
- Updated orchestrator scripts to reuse shared helpers and `ingestion.*` modules.
- Updated ingestion default collection to `sec_docs_hybrid` in `scripts/orchestrate_ingestion.py`.
- Removed hardcoded API key usage from `scripts/download_xbrl.py`; now requires `SEC_API_KEY`.
- Added runtime-boundary audit in `artifacts/scripts_runtime_usage_audit.md`.

### Tracking guidance
- Add one dated section per meaningful change batch.
- Keep entries concise: what changed, why, and impact.
- Reference affected paths for fast navigation.
