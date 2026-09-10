//! 会议录音转写助手 —— Tauri 2 桌面壳。
//!
//! 壳本身不含业务逻辑：它负责
//!   1. 定位并 spawn Python 服务二进制 `asr-server.exe`（作为资源打包，手动启动，
//!      而非 Tauri shell externalBin sidecar 机制），注入 `ASR_SIDECAR=1`；
//!   2. 轮询等待服务在 127.0.0.1:18765 就绪；
//!   3. 创建主窗口加载 http://127.0.0.1:18765/；
//!   4. 退出时杀掉 sidecar 子进程，避免 asr-server.exe 残留。
//!
//! 外围体验：
//!   - 系统托盘：启动即显示；菜单含「显示主窗口」「退出」；左键图标切换显示/聚焦；
//!     点窗口关闭按钮不退出，而是隐藏到托盘，托盘「退出」才真正退出（并 kill sidecar）。
//!   - 单实例：第二次启动聚焦已存在的主窗口，而非开新进程。
//!   - 桌面通知：装好插件 + 权限，提供 `send_notification` 命令供后续 sidecar 事件调用，
//!     启动时不自动弹通知。
//!
//! 生产布局：`resource_dir/asr-server/asr-server.exe`
//! （见 tauri.conf.json 的 bundle.resources 映射）。
//! 开发态回退：仓库根 `dist/asr-server.exe`；都没有则提示 `python app.py`。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent,
};
use tauri_plugin_notification::NotificationExt;

/// sidecar 监听地址（Python 端在 ASR_SIDECAR=1 时锁定此端口）。
const SIDECAR_HOST: &str = "127.0.0.1";
const SIDECAR_PORT: u16 = 18765;
/// 等待服务就绪的最长时间与轮询间隔。
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const POLL_INTERVAL: Duration = Duration::from_millis(300);

// Windows：把 sidecar 整棵进程树挂进 KILL_ON_JOB_CLOSE 作业对象。
// PyInstaller onefile 的 asr-server.exe 是「引导进程→真正的服务进程（孙进程，占 18765）」
// 两层结构，std 的 Child::kill 只杀直接子进程，孙进程会成为孤儿。
#[cfg(windows)]
mod win_job {
    use std::os::windows::io::RawHandle;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    };

    /// KILL_ON_JOB_CLOSE 作业；最后一个句柄关闭（正常退出/崩溃被强杀）时，
    /// 系统自动终止作业内的所有进程及其后代。
    ///
    /// HANDLE 是裸指针默认 !Send，但作业句柄可跨线程使用（CloseHandle/释放可在任意线程），
    /// 且 SidecarState 必须 Send + Sync 才能被 Tauri 管理，这里手动声明 Send。
    pub(super) struct WinJob(HANDLE);
    unsafe impl Send for WinJob {}

    impl WinJob {
        /// 创建作业并把进程挂入（其以后派生的所有子孙默认都在作业内）。失败返回 None。
        pub(super) fn attach(process: RawHandle) -> Option<WinJob> {
            unsafe {
                let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if job.is_null() || job == INVALID_HANDLE_VALUE {
                    return None;
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const std::ffi::c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 {
                    let _ = CloseHandle(job);
                    return None;
                }
                if AssignProcessToJobObject(job, process as HANDLE) == 0 {
                    let _ = CloseHandle(job);
                    return None;
                }
                Some(WinJob(job))
            }
        }
    }

    impl Drop for WinJob {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
}

/// sidecar 进程：子进程句柄 + Windows 作业对象（兜底回收整棵树）。
struct Sidecar {
    child: Child,
    // 不显式读取：它靠 Drop 时关闭句柄触发 KILL_ON_JOB_CLOSE 兜底回收整棵进程树。
    #[cfg(windows)]
    #[allow(dead_code)]
    job: Option<win_job::WinJob>,
}

impl Sidecar {
    /// 终止整棵 sidecar 进程树。先 taskkill /T /F 显式连根杀（确定性），
    /// self 随后被 drop 时 Job Object 句柄关闭再兜底（连崩溃场景也覆盖）。
    fn kill_tree(&mut self) {
        #[cfg(windows)]
        {
            let _ = Command::new("taskkill")
                .args(["/PID", &self.child.id().to_string(), "/T", "/F"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// 持有 sidecar 句柄；应用退出时统一回收整棵进程树。
struct SidecarState(Mutex<Option<Sidecar>>);

impl Drop for SidecarState {
    fn drop(&mut self) {
        kill_sidecar(&mut self.0.lock().unwrap());
    }
}

fn kill_sidecar(slot: &mut Option<Sidecar>) {
    if let Some(mut sc) = slot.take() {
        sc.kill_tree();
        // sc 在此 drop：Windows 上关闭 Job 句柄，KILL_ON_JOB_CLOSE 兜底清掉任何残余后代。
    }
}

/// 显示并聚焦主窗口（若已创建）。托盘菜单 / 单实例回调 / 托盘点击共用。
fn show_main_window(app: &tauri::AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.show();
        let _ = win.unminimize();
        let _ = win.set_focus();
    }
}

/// 左键托盘图标：窗口可见且未最小化则隐藏，否则显示并聚焦。
fn toggle_main_window(app: &tauri::AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        let visible = win.is_visible().unwrap_or(false);
        let minimized = win.is_minimized().unwrap_or(false);
        if visible && !minimized {
            let _ = win.hide();
        } else {
            let _ = win.show();
            let _ = win.unminimize();
            let _ = win.set_focus();
        }
    }
    // 窗口尚未创建（sidecar 还在预热）时无窗口可切换，忽略即可。
}

/// 桌面通知命令：供前端或后续 sidecar 事件（如转写完成、出错）调用。
/// 刻意不在启动时自动弹出，避免每次启动打扰用户。
#[tauri::command]
fn send_notification(app: tauri::AppHandle, title: String, body: String) -> Result<(), String> {
    app.notification()
        .builder()
        .title(&title)
        .body(&body)
        .show()
        .map_err(|e| format!("通知发送失败: {e}"))
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

/// 创建系统托盘图标与菜单。启动即调用，独立于延迟创建的主窗口。
fn build_tray(app: &tauri::AppHandle) -> tauri::Result<()> {
    let show_item = MenuItem::with_id(app, "show", "显示主窗口", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "退出", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show_item, &quit_item])?;

    let mut builder = TrayIconBuilder::with_id("main-tray")
        .tooltip("会议录音转写助手")
        .menu(&menu)
        // 左键留给“切换窗口”，右键才弹菜单（Windows 默认左键也弹菜单，这里关掉）。
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "show" => show_main_window(app),
            "quit" => {
                // 真正退出：先复用 SidecarState 清理逻辑杀掉 sidecar，再退出。
                if let Some(state) = app.try_state::<SidecarState>() {
                    kill_sidecar(&mut state.inner().0.lock().unwrap());
                }
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                toggle_main_window(tray.app_handle());
            }
        });

    // 优先用编译进程序的默认窗口图标（打包/tauri dev 均会嵌入 icons）；
    // 兜底用编译期 include_bytes! 读 icons/icon.png 解码，避免运行期路径依赖。
    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    } else if let Ok(icon) = tauri::image::Image::from_bytes(include_bytes!(
        "../icons/icon.png"
    )) {
        builder = builder.icon(icon);
    }

    builder.build(app)?;
    Ok(())
}

fn main() {
    tauri::Builder::default()
        // 单实例必须最早注册：第二次启动时聚焦已有窗口，而非开新进程。
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            show_main_window(app);
        }))
        .plugin(tauri_plugin_notification::init())
        .invoke_handler(tauri::generate_handler![send_notification])
        .on_window_event(|window, event| {
            // 关闭主窗口 = 隐藏到托盘，不退出应用；托盘「退出」才真正退出。
            if let WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .setup(|app| {
            // 托盘启动即显示（不依赖 sidecar / 主窗口是否就绪）。
            if let Err(e) = build_tray(app.handle()) {
                eprintln!("[shell] failed to create tray icon: {e}");
            }

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
                    // Windows：把 sidecar（及其以后派生的全部子孙）挂进 KILL_ON_JOB_CLOSE 作业。
                    #[cfg(windows)]
                    let job = {
                        use std::os::windows::io::AsRawHandle;
                        win_job::WinJob::attach(child.as_raw_handle())
                    };
                    #[cfg(not(windows))]
                    let job = ();
                    let _ = &job;
                    app.manage(SidecarState(Mutex::new(Some(Sidecar {
                        child,
                        #[cfg(windows)]
                        job,
                    }))));
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
                    kill_sidecar(&mut state.inner().0.lock().unwrap());
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
