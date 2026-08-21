# 🎙️ 会议录音转写助手 v3.1.0

自动检测腾讯会议启动，录制系统音频并转写为带时间戳的文字稿，支持 AI 摘要与 Word 导出。

> **架构**：Python 后端（FastAPI + WebSocket）+ 浏览器前端（Edge app 模式，无地址栏，像原生应用）

## ✨ 功能特点

- **自动检测** — 监控腾讯会议进程，开会自动录音，结束自动停止
- **系统音频捕获** — 通过「立体声混音」直接录制电脑播放的声音，无需麦克风
- **本地语音转写** — 默认使用 FunASR `paraformer-large-vad-punc`（中文识别准、CPU RTF 低），可在设置中切换为 `SenseVoice`，模型下载后可离线运行
- **实时转写** — 可选讯飞 WebSocket 实时听写
- **AI 摘要** — 支持讯飞 / 火山引擎 / DeepSeek / 通义千问
- **文档导出** — 导出为 txt 或 Word（.docx）

## 🚀 快速开始

### 环境要求

- **Python 3.11**（`requirements.txt` 按 3.11 验证）
- Windows（依赖「立体声混音」录制系统音频）
- Microsoft Edge（作为界面容器；缺失时回退到默认浏览器）

### 1. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

或使用 [uv](https://github.com/astral-sh/uv)（更快）：

```powershell
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

> 首次安装约需下载 2-3 GB（torch / torchaudio 等），请耐心等待。

### 2. 启动程序

双击 `start.bat`，或手动运行：

```powershell
$env:PYTHONIOENCODING = "utf-8"    # Windows 必须，否则中文设备名会触发 GBK 编码错误
.venv\Scripts\python.exe app.py
```

程序会自动选择一个空闲端口并用 Edge 打开界面，控制台会打印实际地址，例如：

```
启动 会议录音转写助手 v3.1.0
访问地址: http://127.0.0.1:18766/
```

> ⚠️ 端口是**每次启动动态分配**的，不是固定值。

### 3. 首次启动：自动下载语音模型

首次运行会从 [ModelScope](https://modelscope.cn) 自动下载模型到 `~/.cache/modelscope`，共约 **2.9 GB**：

| 模型 | 大小 | 用途 |
|------|------|------|
| `iic/punc_ct-transformer_cn-en-common-vocab471067-large` | 1.13 GB | 标点恢复 |
| `iic/SenseVoiceSmall` | 897 MB | 文件转写 |
| `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` | 848 MB | 实时转写 |
| `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 3.8 MB | 语音端点检测（VAD） |

下载完成后即可离线使用。

> **⚠️ 重要**：模型在后台守护线程中加载，**若下载失败不会有任何弹窗或报错**，
> 表现为界面能打开、能录音，但点击转写毫无反应。
>
> 排查方法：检查 `~/.cache/modelscope/models` 是否存在且包含上述模型目录。
> 若为空，请检查网络后**在前台运行 `python app.py` 观察下载日志**。

## 🔊 系统音频设置

录制会议声音需要启用「立体声混音」。先确认是否已可用：

```powershell
.venv\Scripts\python.exe -c "import sounddevice; print(sounddevice.query_devices())"
```

输出中若存在「立体声混音」/「Stereo input」条目即为正常，例如：

```
id=13  立体声混音 (Realtek HD Audio Stereo input)  ch=2  sr=48000
```

若不存在，手动启用：

1. 右键点击右下角音量图标 → 「声音设置」
2. 点击「更多声音设置」（或「声音控制面板」）
3. 切换到「录制」标签
4. 右键空白处，勾选「显示已禁用的设备」
5. 找到「立体声混音」，右键启用并设为默认

**替代方案**：安装 [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)（免费虚拟音频线）

## 🎛️ 使用方式

- **自动模式** — 程序自动检测腾讯会议启停并录音
- **手动模式** — 随时手动开始 / 停止录音
- **文件转写** — 选择已有音频文件进行转写

音频来源可选：系统声音 / 麦克风 / 两者混合（默认 `both`）。

## ⚙️ 可选配置

以下功能需在界面中填入相应密钥，**不配置也能正常使用本地转写**：

| 功能 | 所需配置 |
|------|----------|
| 讯飞实时转写 | `xfyun_app_id`、`xfyun_api_key`、`xfyun_api_secret` |
| AI 摘要 | `llm_api_key`、`llm_provider`（`xfyun` / `volcengine` / `deepseek` / `qwen`） |
| 热词优化 | `hot_words`（提升专有名词识别准确率） |

## 📁 文件存储

```
~/MeetingRecorder/
├── recordings/          # 录音文件
├── transcripts/         # 转写结果
└── startup_error.log    # 启动错误日志
```

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/api/config` | 读取当前配置 |
| GET | `/api/devices` | 列出可用音频输入设备 |
| WS | `/ws` | 录音控制、实时转写、状态推送 |

## 📦 打包

使用 PyInstaller 按 `meeting-recorder.spec` 打包：

```powershell
.venv\Scripts\python.exe -m PyInstaller meeting-recorder.spec
```

> 打包产物**不包含语音模型**，首次运行仍会下载到 `~/.cache/modelscope`。

## 🐛 常见问题

**界面没有出现？**
程序以 Edge app 模式（无地址栏）启动，窗口可能被其他窗口遮挡。
从控制台输出的地址手动打开即可。

**点击转写没反应？**
语音模型未下载成功。见上文「首次启动：自动下载语音模型」。

**录不到会议声音？**
「立体声混音」未启用。见上文「系统音频设置」。

**中文输出乱码或 `UnicodeEncodeError: 'gbk' codec`？**
运行前设置 `$env:PYTHONIOENCODING = "utf-8"`（`start.bat` 已内置）。

## 📋 主要依赖

完整列表见 [`requirements.txt`](requirements.txt)。

- **Web** — fastapi、uvicorn
- **音频** — sounddevice、numpy
- **转写** — funasr、modelscope、torch、torchaudio
- **实时转写** — websocket-client
- **导出** — python-docx
