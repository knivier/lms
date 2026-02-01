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

/// State for the cv.py native feed: Python child process so we can kill it on stop.
struct CvFeedState {
    child: Mutex<Option<Child>>,
}

/// Resolve path to cv_stdout_frames.py (repo/cv/cv_stdout_frames.py relative to crate).
fn cv_stdout_frames_path() -> Result<std::path::PathBuf, String> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    // Crate is front/src-tauri, so repo root is manifest_dir/../..
    let script = manifest_dir.join("../../cv/cv_stdout_frames.py");
    script
        .canonicalize()
        .map_err(|e| format!("cv_stdout_frames.py not found at {}: {}", script.display(), e))
}

/// Start the cv.py pipeline (camera + skeleton); frames are emitted as "cv-frame" (base64 JPEG string).
#[tauri::command]
fn start_cv_feed(app: tauri::AppHandle, state: tauri::State<'_, CvFeedState>) -> Result<(), String> {
    // Don't start twice
    {
        let mut guard = state.child.lock().map_err(|e| e.to_string())?;
        if guard.is_some() {
            return Ok(());
        }
    }

    let script_path = cv_stdout_frames_path()?;
    let mut child = Command::new("python3")
        .arg(&script_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .or_else(|_| {
            Command::new("python")
                .arg(&script_path)
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
        let mut guard = state.child.lock().map_err(|e| e.to_string())?;
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

    Ok(())
}

/// Stop the cv.py pipeline (kill the Python process).
#[tauri::command]
fn stop_cv_feed(state: tauri::State<'_, CvFeedState>) -> Result<(), String> {
    let mut guard = state.child.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
    }
    Ok(())
}

/// Write the selected workout id to workout_id.json (one line: {"workout_id":"squat"}).
#[tauri::command]
fn write_workout_id(workout_id: String) -> Result<(), String> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .join("../..")
        .canonicalize()
        .map_err(|e| format!("repo root: {}", e))?;
    let path = repo_root.join("workout_id.json");
    let line = serde_json::json!({ "workout_id": workout_id }).to_string();
    std::fs::write(&path, line).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(CvFeedState {
            child: Mutex::new(None),
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
