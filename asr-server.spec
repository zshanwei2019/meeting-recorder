# -*- mode: python ; coding: utf-8 -*-
"""
Release 打包配置：onefile 模式输出单个 asr-server.exe。
Tauri sidecar.rs 在 resource_dir/asr-server/asr-server.exe 找它。
onefile 启动时自解压到 %TEMP%，避免 Tauri/NSIS 打包目录结构问题。
用法: pyinstaller asr-server.spec
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

funasr_datas = collect_data_files('funasr')
modelscope_datas = collect_data_files('modelscope')
funasr_hiddenimports = collect_submodules('funasr')
modelscope_hiddenimports = collect_submodules('modelscope')

_pyannote_pkgs = [
    'pyannote.audio', 'pyannote.core', 'pyannote.database',
    'pyannote.pipeline', 'pyannote.metrics',
    'hydra', 'omegaconf', 'asteroid_filterbanks',
]
pyannote_datas = []
pyannote_hiddenimports = []
for _pkg in _pyannote_pkgs:
    try:
        pyannote_datas += collect_data_files(_pkg)
    except Exception:
        pass
    try:
        pyannote_hiddenimports += collect_submodules(_pkg)
    except Exception:
        pass

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('ui/index.html', 'ui'),
        ('app_icon.ico', '.'),
        ('app_icon.png', '.'),
    ] + funasr_datas + modelscope_datas + pyannote_datas,
    hiddenimports=[
        'pyannote_chunk_worker',
        'pyannote.audio.pipelines.speaker_diarization',
        'soundfile',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'sounddevice',
        'numpy',
        'scipy._lib.messagestream',
        'websockets',
        'websocket',
        'websocket_client',
    ] + funasr_hiddenimports + modelscope_hiddenimports + pyannote_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'tkinter', 'PIL', 'scipy.spatial',
        'scipy.ndimage', 'IPython', 'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='asr-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    icon='app_icon.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
