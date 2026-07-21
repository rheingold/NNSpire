/// Storage layer for NNSpire Agent chat state.
///
/// Persists chat conversations and folders to JSON files on disk.
/// Storage location follows ADR-052: %APPDATA%/NNSpire/nnagent/ on Windows,
/// ~/.nnspire/nnagent/ on Linux/macOS.
///
/// The state is stored as a single `chat_state.json` file for now.
/// Future evolution will split into per-conversation files.

use std::fs;
use std::path::{Path, PathBuf};

/// Root storage directory for the application.
/// Returns `%APPDATA%/NNSpire/nnagent` on Windows or `~/.nnspire/nnagent` on Unix.
fn get_storage_dir() -> Result<PathBuf, String> {
    let base_dir = dirs::data_dir().ok_or_else(|| {
        "[storage] Unable to determine application data directory".to_string()
    })?;
    
    let nnspire_dir = base_dir.join("NNSpire").join("nnagent");
    Ok(nnspire_dir)
}

/// Ensure the storage directory exists, creating it if necessary.
fn ensure_storage_dir() -> Result<PathBuf, String> {
    let dir = get_storage_dir()?;
    fs::create_dir_all(&dir).map_err(|e| {
        format!("[storage] Failed to create storage directory {:?}: {}", dir, e)
    })?;
    Ok(dir)
}

/// Path to the main chat state file.
fn chat_state_path(storage_dir: &Path) -> PathBuf {
    storage_dir.join("chat_state.json")
}

/// Serialize and write the chat state to disk.
///
/// # Arguments
/// * `state_json` - The serialized JSON string of the chat state.
///
/// # Errors
/// Returns a `String` error message if writing fails.
pub fn save_chat_state(state_json: &str) -> Result<(), String> {
    let storage_dir = ensure_storage_dir()?;
    let file_path = chat_state_path(&storage_dir);
    
    // Write to a temporary file first, then rename for atomicity
    let temp_path = storage_dir.join("chat_state.json.tmp");
    
    fs::write(&temp_path, state_json).map_err(|e| {
        format!("[storage] Failed to write chat state to {:?}: {}", temp_path, e)
    })?;
    
    // Rename is atomic on most filesystems
    fs::rename(&temp_path, &file_path).map_err(|e| {
        format!("[storage] Failed to finalize chat state file {:?}: {}", file_path, e)
    })?;
    
    log::info!("[storage] Chat state saved successfully to {:?}", file_path);
    Ok(())
}

/// Read and deserialize the chat state from disk.
///
/// # Returns
/// The serialized JSON string of the chat state, or `None` if no state file exists.
///
/// # Errors
/// Returns a `String` error message if reading fails.
pub fn load_chat_state() -> Result<Option<String>, String> {
    let storage_dir = match get_storage_dir() {
        Ok(dir) => dir,
        Err(e) => return Err(e),
    };
    
    let file_path = chat_state_path(&storage_dir);
    
    // If file doesn't exist, return None (no state to load)
    if !file_path.exists() {
        log::info!("[storage] No chat state file found at {:?} — returning empty state", file_path);
        return Ok(None);
    }
    
    let content = fs::read_to_string(&file_path).map_err(|e| {
        format!("[storage] Failed to read chat state from {:?}: {}", file_path, e)
    })?;
    
    log::info!("[storage] Chat state loaded successfully from {:?}", file_path);
    Ok(Some(content))
}

/// Migrate state from localStorage marker.
///
/// This is a placeholder for future migration logic. When the app first runs
/// with file-based storage, it will check for a localStorage migration flag.
/// The actual migration is handled from the frontend side, as Rust cannot
/// access WebView2 localStorage directly.
pub fn mark_migration_complete() -> Result<(), String> {
    let storage_dir = ensure_storage_dir()?;
    let migration_flag = storage_dir.join(".localStorage_migrated");
    
    fs::write(&migration_flag, "migrated").map_err(|e| {
        format!("[storage] Failed to write migration flag to {:?}: {}", migration_flag, e)
    })?;
    
    log::info!("[storage] LocalStorage migration marked as complete");
    Ok(())
}

/// Check if localStorage migration has been completed.
pub fn is_migration_complete() -> bool {
    let storage_dir = match get_storage_dir() {
        Ok(dir) => dir,
        Err(_) => return true, // If we can't get the dir, assume migrated
    };
    let migration_flag = storage_dir.join(".localStorage_migrated");
    migration_flag.exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_storage_dir_returns_valid_path() {
        let dir = get_storage_dir();
        assert!(dir.is_ok(), "get_storage_dir should return a valid path");
        let path = dir.unwrap();
        assert!(path.ends_with("nnagent") || path.to_string_lossy().contains("nnagent"));
    }

    #[test]
    fn test_chat_state_path_is_correct() {
        let storage_dir = std::path::PathBuf::from("/tmp/test_storage");
        let expected = storage_dir.join("chat_state.json");
        let actual = chat_state_path(&storage_dir);
        assert_eq!(actual, expected);
    }
}
