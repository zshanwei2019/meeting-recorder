# 桌面壳打包说明（Tauri 2 + Python sidecar）

「会议录音转写助手」桌面版 = 一个很轻的 **Tauri 2 (Rust) 壳** + 一个用
PyInstaller 打成单文件的 **Python 服务 `asr-server.exe`**（FastAPI + uvicorn，
约 400MB，含 ASR 模型依赖）。

壳本身不含业务逻辑：启动时把 `asr-server.exe` 当**资源**（不是 Tauri shell
externalBin sidecar 机制）手动 spawn，注入环境变量 `ASR_SIDECAR=1`，等服务在
`127.0.0.1:18765` 就绪后，用 WebView 窗口加载 `http://127.0.0.1:18765/`；
退出时杀掉子进程。

## 关键标识

| 项 | 值 |
|---|---|
| productName / 可执行名 | `NewMeetingRecorder`（`NewMeetingRecorder.exe`） |
| identifier | `com.newmeetingrecorder.app` |
| 窗口标题 | 会议录音转写助手 |
| sidecar 资源路径（安装后） | `<resource_dir>/asr-server/asr-server.exe` |
| sidecar 监听 | `127.0.0.1:18765`（`ASR_SIDECAR=1` 时锁定） |
| 安装包类型 | NSIS（`installMode: currentUser`，免管理员） |

## 版本号（单一来源）

- 基础版本号只有一处：`version.py` 的 `BASE_VERSION`（当前 `3.1.0`）。
- `tools/gen_version.py`（PyInstaller 打包前由 `asr-server.spec` 自动调用）除了
  生成 `_build_info.py` / `_version.py`，还会把 `BASE_VERSION` **同步写进
  `src-tauri/tauri.conf.json` 的 `version` 字段**。
- Windows 下壳 exe 的**文件版本资源**和 **NSIS 产品版本**都由
  `tauri.conf.json` 的 `version` 驱动 —— 这修复了过去右键属性版本为空 / 0.1.0
  的问题。
- App 内显示的 commit / 构建日期版本仍由 Python `/api/version` 提供
  （`version_string()`），不塞进 Windows 数字版本。

> 发版流程：改 `version.py` 的 `BASE_VERSION` → 跑下面的流水线，
> tauri.conf.json 会被自动同步（该改动会出现在打包工作区，提交时一并入库即可）。

## 完整发布流水线

前置：Rust（MSVC 工具链，`x86_64-pc-windows-msvc`）、Node.js + npm。

```powershell
# 0) （仅首次）安装 Tauri CLI —— 已在根 package.json 的 devDependencies，
#    注意本机 npm 全局可能配置了 omit=dev，需显式带上 dev 依赖：
npm install --include=dev

# 1) 构建 Python 服务（onefile，产物 dist/asr-server.exe，约 405MB）
#    spec 会自动调用 tools/gen_version.py 生成版本信息并同步 tauri 版本
pyinstaller asr-server.spec

# 2) 生成/刷新图标（源：仓库根 app_icon.png；一般无需每次跑）
npx tauri icon app_icon.png

# 3) 构建桌面壳 + NSIS 安装包
#    bundle.resources 会把 dist/asr-server.exe 映射进 asr-server/asr-server.exe
npx tauri build
```

产物：

- 壳 exe：`src-tauri/target/release/NewMeetingRecorder.exe`
- NSIS 安装包：`src-tauri/target/release/bundle/nsis/*.exe`

安装后布局（资源随安装目录释放）：

```
<install-dir>/
  NewMeetingRecorder.exe
  resources/asr-server/asr-server.exe   # ← 壳在这里找服务
```

> 实际 resource_dir 由 Tauri 决定（NSIS currentUser 下通常为
> `<install-dir>/resources/`）；壳用 `app.path().resource_dir()` 拼接
> `asr-server/asr-server.exe`，与 `tauri.conf.json` 的 `bundle.resources`
> 映射（`"../dist/asr-server.exe": "asr-server/asr-server.exe"`）一致。

## 只验证壳编译（不打 405MB 包）

```powershell
# 快速类型/编译检查（不产出安装包）
cd src-tauri; cargo check

# 或编译壳 exe 但不 bundle（不要求 asr-server.exe 存在）
npx tauri build --no-bundle
```

## 开发态

- 壳在找不到打包资源时，会回退查找仓库根 `dist/asr-server.exe`；
- 都没有时仍会建窗，并在控制台打印
  `ASR server not found. Package asr-server.exe or run from development environment`。
  此时可直接在仓库根 `python app.py`（非 sidecar 模式，自选端口）用浏览器开发前端。

## 目录结构

```
src-tauri/
  Cargo.toml
  build.rs
  tauri.conf.json        # productName/identifier/version/bundle/resources/nsis
  capabilities/default.json  # Tauri 2 最小权限（用 std::process，无需 shell 插件）
  icons/                 # 由 `npx tauri icon` 生成
  src/main.rs            # 壳逻辑：定位/spawn sidecar、轮询就绪、建窗、退出清理
  PACKAGING.md           # 本文件
package.json             # 仅 devDependency @tauri-apps/cli + tauri scripts
```
