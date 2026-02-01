# lms

## Session launcher (Tauri app)

The Rust/Tauri app is the **session orchestrator**: when you start a workout it starts the **CV pipeline** (camera + skeleton) and any **session scripts** (e.g. data cleaning). CV output goes into the app; other scripts run in the background.

- **CV**: Started automatically; its stdout (base64 JPEG frames) is read by the app and shown in the Live Session UI.
- **Session scripts**: Listed in **`session_config.json`** at the repo root. Each entry is a command line (e.g. `python ProcessedData/synthesizer.py`). Commands are run from the repo root when the session starts and are stopped when you stop the feed.

Example `session_config.json`:

```json
{
  "session_scripts": [
    "python ProcessedData/synthesizer.py",
    "python quantprocess/RepTracker.py"
  ]
}
```

Leave `session_scripts` empty `[]` to only run CV.