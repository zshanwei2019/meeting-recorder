//! 会议录音转写助手 —— Tauri 2 桌面壳。
//!
//! 壳本身不含业务逻辑：它负责
//!   1. 定位并 spawn Python 服务二进制 `asr-server.exe`（作为资源打包，手动启动，
//!      而非 Tauri shell externalBin sidecar 机制），注入 `ASR_SIDECAR=1`；
//!   2. 轮询等待服务在 127.0.0.1:18765 就绪；
//!   3. 创建主窗口加载 http://127.0.0.1:18765/；
//!   4. 退出时杀掉 sidecar 子进程，避免 asr-server.exe 残留。
//!
//! 生产布局：`resource_dir/asr-server/asr-server.exe`
//! （见 tauri.conf.json 的 bundle.resources 映射）。
//! 开发态回退：仓库根 `dist/asr-server.exe`；都没有则提示 `python app.py`。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

/// sidecar 监听地址（Python 端在 ASR_SIDECAR=1 时锁定此端口）。
const SIDECAR_HOST: &str = "127.0.0.1";
const SIDECAR_PORT: u16 = 18765;
/// 等待服务就绪的最长时间与轮询间隔。
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const POLL_INTERVAL: Duration = Duration::from_millis(300);

/// 持有 sidecar 子进程句柄；应用退出时统一 kill。
struct SidecarState(Mutex<Option<Child>>);

impl Drop for SidecarState {
    fn drop(&mut self) {
        kill_sidecar(&mut self.0.lock().unwrap());
    }
}

fn kill_sidecar(slot: &mut Option<Child>) {
    if let Some(mut child) = slot.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// 定位 asr-server.exe。生产：resource_dir/asr-server/asr-server.exe；
/// 开发：CARGO_MANIFEST_DIR/../dist/asr-server.exe 或当前工作目录附近。
fn locate_sidecar(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    // 1) 生产：打包资源目录。
    if let Ok(res) = app.path().resource_dir() {
        candidates.push(res.join("asr-server").join("asr-server.exe"));
    }

    // 2) 开发态：编译期清单目录（src-tauri）上一级的 dist/。
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest.join("..").join("dist").join("asr-server.exe"));

    // 3) 开发态：当前工作目录（tauri dev 常在仓库根或 src-tauri 启动）。
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("dist").join("asr-server.exe"));
        candidates.push(cwd.join("..").join("dist").join("asr-server.exe"));
    }

    candidates.into_iter().find(|p| p.is_file())
}

/// 启动 sidecar，注入 ASR_SIDECAR=1；stdout/stderr 继承到控制台便于排查。
fn spawn_sidecar(exe: &std::path::Path) -> std::io::Result<Child> {
    Command::new(exe)
        .env("ASR_SIDECAR", "1")
        .current_dir(exe.parent().unwrap_or_else(|| std::path::Path::new(".")))
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
}

/// 对服务发一个最小 HTTP GET，能连上且返回任意响应即视为就绪。
fn http_ready() -> bool {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    let addr = format!("{SIDECAR_HOST}:{SIDECAR_PORT}");
    let Ok(mut stream) = TcpStream::connect_timeout(
        &addr.parse().unwrap(),
        Duration::from_millis(500),
    ) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let req = format!("GET / HTTP/1.0\r\nHost: {addr}\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 128];
    match stream.read(&mut buf) {
        Ok(n) if n > 0 => buf.starts_with(b"HTTP/"),
        _ => false,
    }
}

/// 轮询直到服务就绪或超时。
fn wait_until_ready(deadline: Instant) -> bool {
    while Instant::now() < deadline {
        if http_ready() {
            return true;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    false
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // 定位并启动 sidecar。
            let Some(exe) = locate_sidecar(app.handle()) else {
                // 开发态没有打包资源、也没构建 dist 时的友好提示。
                eprintln!(
                    "ASR server not found. Package asr-server.exe or run from development environment"
                );
                eprintln!(
                    "  - 生产：应为 <resource_dir>/asr-server/asr-server.exe"
                );
                eprintln!("  - 开发：先 `pyinstaller asr-server.spec` 生成 dist/asr-server.exe，");
                eprintln!("         或直接在仓库根 `python app.py` 后用浏览器访问。");
                // 仍然建窗，指向目标地址（服务不在时窗口会显示无法访问，便于排查）。
                build_main_window(app.handle())?;
                app.manage(SidecarState(Mutex::new(None)));
                return Ok(());
            };

            println!("[shell] launching sidecar: {}", exe.display());
            match spawn_sidecar(&exe) {
                Ok(child) => {
                    app.manage(SidecarState(Mutex::new(Some(child))));
                }
                Err(e) => {
                    eprintln!("[shell] failed to spawn sidecar: {e}");
                    app.manage(SidecarState(Mutex::new(None)));
                }
            }

            // 等待服务就绪（后台线程，避免阻塞 setup；就绪后再建窗）。
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let ready = wait_until_ready(Instant::now() + READY_TIMEOUT);
                if !ready {
                    eprintln!(
                        "[shell] sidecar did not become ready within {}s",
                        READY_TIMEOUT.as_secs()
                    );
                }
                // 在主线程创建窗口（克隆 handle 移入闭包，避免借用/移动冲突）。
                let win_handle = handle.clone();
                let _ = handle.run_on_main_thread(move || {
                    if let Err(e) = build_main_window(&win_handle) {
                        eprintln!("[shell] failed to create window: {e}");
                    }
                });
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // 应用退出时杀掉 sidecar，避免 asr-server.exe 残留。
            if let RunEvent::Exit | RunEvent::ExitRequested { .. } = event {
                if let Some(state) = app_handle.try_state::<SidecarState>() {
                    kill_sidecar(&mut state.0.lock().unwrap());
                }
            }
        });
}

/// 创建主窗口，加载 sidecar 提供的页面。
fn build_main_window(app: &tauri::AppHandle) -> tauri::Result<tauri::WebviewWindow> {
    let url = format!("http://{SIDECAR_HOST}:{SIDECAR_PORT}/")
        .parse()
        .expect("valid sidecar url");
    WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
        .title("会议录音转写助手")
        .inner_size(1280.0, 860.0)
        .min_inner_size(1000.0, 700.0)
        .resizable(true)
        .build()
}
