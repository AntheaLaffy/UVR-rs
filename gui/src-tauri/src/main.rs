#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod locale;
mod models;
mod runtime;
mod tasks;
mod version;

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(tasks::TaskState::default())
        .manage(models::LibraryState::default())
        .invoke_handler(tauri::generate_handler![
            tasks::start_task,
            tasks::cancel_task,
            tasks::choose_path,
            tasks::defaults,
            models::inspect_models,
            models::download_model,
            version::check_update
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event
                && window.state::<tasks::TaskState>().cancel_for_close()
            {
                api.prevent_close();
            }
        })
        .run(tauri::generate_context!())
        .expect("无法启动 UVR 桌面应用");
}
