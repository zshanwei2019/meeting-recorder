# -*- mode: python ; coding: utf-8 -*-
"""
会议录音转写助手 v3.1.0 - PyInstaller 打包配置
用法: pyinstaller meeting-recorder.spec
"""
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 收集 funasr / modelscope 的数据文件（模型配置等）
funasr_datas = collect_data_files('funasr')
modelscope_datas = collect_data_files('modelscope')

# 收集 funasr 的所有子模块（动态导入）
funasr_hiddenimports = collect_submodules('funasr')
modelscope_hiddenimports = collect_submodules('modelscope')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('ui/index.html', 'ui'),          # 前端页面
        ('app_icon.ico', '.'),             # 应用图标
        ('app_icon.png', '.'),             # 应用图标PNG
    ] + funasr_datas + modelscope_datas,
    hiddenimports=[
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
    ] + funasr_hiddenimports + modelscope_hiddenimports,
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
