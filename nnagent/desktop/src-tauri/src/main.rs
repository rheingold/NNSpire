// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod storage;

/// Tauri command: Save chat state to disk.
///
/// Receives the serialized chat state as a JSON string and persists it to
/// the application data directory.
#[tauri::command]
fn cmd_save_chat_state(state_json: String) -> Result<(), String> {
    storage::save_chat_state(&state_json)
}

/// Tauri command: Load chat state from disk.
///
/// Returns the serialized chat state JSON string, or null if no state exists.
#[tauri::command]
fn cmd_load_chat_state() -> Result<Option<String>, String> {
    storage::load_chat_state()
}

/// Tauri command: Mark localStorage migration as complete.
///
/// Called by the frontend after successfully migrating data from localStorage
/// to file-based storage.
#[tauri::command]
fn cmd_mark_migration_complete() -> Result<(), String> {
    storage::mark_migration_complete()
}

/// Tauri command: Check if localStorage migration is complete.
#[tauri::command]
fn cmd_is_migration_complete() -> bool {
    storage::is_migration_complete()
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::default().build())
        .invoke_handler(tauri::generate_handler![
            cmd_save_chat_state,
            cmd_load_chat_state,
            cmd_mark_migration_complete,
            cmd_is_migration_complete
        ])
        .run(tauri::generate_context!())
        .expect("error while running NNSpire Agent");
}
