# -*- mode: python ; coding: utf-8 -*-
"""
会议录音转写助手 v3.1.0 - PyInstaller 打包配置
用法: pyinstaller meeting-recorder.spec
"""
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 打包前自动生成 _version.py（固化 git commit/日期进产物）
import subprocess
try:
    subprocess.run([sys.executable, os.path.join(SPECPATH, 'tools', 'gen_version.py')],
                   cwd=SPECPATH, check=False)
except Exception:
    pass
_version_data = [('_version.py', '.')] if os.path.exists(os.path.join(SPECPATH, '_version.py')) else []

block_cipher = None

# 收集 funasr / modelscope 的数据文件（模型配置等）
funasr_datas = collect_data_files('funasr')
modelscope_datas = collect_data_files('modelscope')

# 收集 funasr 的所有子模块（动态导入）
funasr_hiddenimports = collect_submodules('funasr')
modelscope_hiddenimports = collect_submodules('modelscope')

# pyannote 说话人分离：大量动态实例化（hydra/omegaconf 通过配置字符串 import），
# 需把 pyannote 全家桶 + hydra + asteroid_filterbanks 的子模块和数据全收进来。
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
        ('ui/index.html', 'ui'),          # 前端页面
        ('app_icon.ico', '.'),             # 应用图标
        ('app_icon.png', '.'),             # 应用图标PNG
    ] + _version_data + funasr_datas + modelscope_datas + pyannote_datas,
    hiddenimports=[
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
    [],
    exclude_binaries=True,
    name='会议录音转写助手',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,            # 显示命令行窗口，便于查看错误日志
    icon='app_icon.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='会议录音转写助手',
)
