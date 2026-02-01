// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::Emitter;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Session state: CV process (stdout → app) + any session scripts (e.g. data cleaning) started from session_config.json.
struct SessionState {
    cv_child: Mutex<Option<Child>>,
    session_script_children: Mutex<Vec<Child>>,
}

fn repo_root() -> Result<std::path::PathBuf, String> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .join("../..")
        .canonicalize()
        .map_err(|e| format!("repo root: {}", e))
}

/// Resolve path to cv_stdout_frames.py (repo/cv/cv_stdout_frames.py relative to crate).
fn cv_stdout_frames_path() -> Result<std::path::PathBuf, String> {
    let root = repo_root()?;
    let script = root.join("cv/cv_stdout_frames.py");
    script
        .canonicalize()
        .map_err(|e| format!("cv_stdout_frames.py not found at {}: {}", script.display(), e))
}

/// Start CV pipeline (output → app) and any session_scripts from session_config.json (repo root).
#[tauri::command]
fn start_cv_feed(app: tauri::AppHandle, state: tauri::State<'_, SessionState>) -> Result<(), String> {
    // Don't start twice
    {
        let mut guard = state.cv_child.lock().map_err(|e| e.to_string())?;
        if guard.is_some() {
            return Ok(());
        }
    }

    let root = repo_root()?;
    let script_path = cv_stdout_frames_path()?;
    let mut child = Command::new("python3")
        .arg(&script_path)
        .current_dir(&root)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .or_else(|_| {
            Command::new("python")
                .arg(&script_path)
                .current_dir(&root)
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
        })
        .map_err(|e| format!("Failed to run cv.py pipeline: {}", e))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "No stdout from cv process".to_string())?;

    {
        let mut guard = state.cv_child.lock().map_err(|e| e.to_string())?;
        *guard = Some(child);
    }

    let app = app.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(b64) => {
                    let _ = app.emit("cv-frame", &b64);
                }
                Err(_) => break,
            }
        }
    });

    // Start session scripts from session_config.json (e.g. data cleaning)
    let config_path = root.join("session_config.json");
    if let Ok(buf) = std::fs::read_to_string(&config_path) {
        if let Ok(cfg) = serde_json::from_str::<SessionConfig>(&buf) {
            let mut children = state.session_script_children.lock().map_err(|e| e.to_string())?;
            for cmd in cfg.session_scripts {
                if cmd.is_empty() {
                    continue;
                }
                // Run first token as program, rest as args (e.g. "python ProcessedData/synthesizer.py")
                let parts: Vec<&str> = cmd.split_whitespace().collect();
                if parts.is_empty() {
                    continue;
                }
                let mut c = Command::new(parts[0])
                    .args(parts.iter().skip(1))
                    .current_dir(&root)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn();
                if let Ok(proc) = c {
                    children.push(proc);
                }
            }
        }
    }

    Ok(())
}

#[derive(serde::Deserialize)]
struct SessionConfig {
    #[serde(default)]
    session_scripts: Vec<String>,
}

/// Stop CV pipeline and all session scripts.
#[tauri::command]
fn stop_cv_feed(state: tauri::State<'_, SessionState>) -> Result<(), String> {
    {
        let mut guard = state.cv_child.lock().map_err(|e| e.to_string())?;
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
        }
    }
    let mut guard = state.session_script_children.lock().map_err(|e| e.to_string())?;
    for mut child in guard.drain(..) {
        let _ = child.kill();
    }
    Ok(())
}

/// Write workout_id.json: line 1 = {"workout_id":"squat"}, line 2 = "ON" or "OFF" (session active).
#[tauri::command]
fn write_workout_id(workout_id: String, session: String) -> Result<(), String> {
    let root = repo_root()?;
    let path = root.join("workout_id.json");
    let line1 = serde_json::json!({ "workout_id": workout_id }).to_string();
    let line2 = if session.eq_ignore_ascii_case("on") { "ON" } else { "OFF" };
    let content = format!("{}\n{}\n", line1, line2);
    std::fs::write(&path, content).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(SessionState {
            cv_child: Mutex::new(None),
            session_script_children: Mutex::new(Vec::new()),
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            start_cv_feed,
            stop_cv_feed,
            write_workout_id,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
