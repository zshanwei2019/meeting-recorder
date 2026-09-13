# -*- coding: utf-8 -*-
"""统一运行日志：写文件（按大小轮转）+ 可选控制台。

为什么需要它：此前全项目用 print，打包成无控制台 GUI 后日志几乎全部丢失，
长录音/云端转写出问题只能凭用户截图里的一句模糊提示排查。本模块提供：
- ~/MeetingRecorder/logs/app.log，RotatingFileHandler 单文件 5MB × 保留 5 份
- 侧车（ASR_SIDECAR=1）/frozen 环境下 stdout 被重定向到 NUL，自动只留文件 handler
- 第三方嘈杂库（urllib3 等）压到 WARNING，业务日志走 "meeting_recorder" logger
- setup_logging() 幂等，可重复调用不重复挂 handler
- 纯标准库，模块导入零副作用（不调用就不建文件）

测试可通过 setup_logging(log_dir=临时目录) 注入路径，或直接断言 handler 配置。
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

LOGGER_NAME = "meeting_recorder"
DEFAULT_LOG_DIR = Path.home() / "MeetingRecorder" / "logs"
DEFAULT_LOG_FILE = "app.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5
LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def is_headless() -> bool:
    """frozen sidecar 或 stdout 被重定向到空设备时，控制台输出无意义。"""
    if getattr(sys, "frozen", False):
        return True
    if os.environ.get("ASR_SIDECAR", "0") == "1":
        return True
    try:
        return not sys.stdout or not sys.stdout.isatty()
    except Exception:
        return False


def setup_logging(log_dir=None, *, level=logging.INFO, console=None, file_log=True):
    """初始化全局 logger，幂等。返回 logger。

    log_dir: 日志目录（默认 ~/MeetingRecorder/logs）；
    console: None=自动（有 TTY 才加控制台 handler），True/False 强制；
    file_log: False 时只配置级别不加文件 handler（测试用）。
    """
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if _configured:
        return logger

    formatter = logging.Formatter(LOG_FMT, datefmt=DATE_FMT)

    if file_log:
        try:
            directory = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
            directory.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                directory / DEFAULT_LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:  # 日志失败绝不能拖垮主程序
            print(f"[WARN] 文件日志初始化失败: {e}", file=sys.stderr)

    want_console = (not is_headless()) if console is None else console
    if want_console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = False  # 不冒泡到 root，避免重复/被第三方配置影响
    _configured = True

    # 压住高频第三方库的调试噪音（requests 连接池等）
    for noisy in ("urllib3", "websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def get_logger(name: str = "") -> logging.Logger:
    """取业务 logger；子模块用 get_logger('cloud_asr') 等。

    未初始化时返回一个配置了 NullHandler 的 logger，保证 import 期/单测里
    调用不报错、也不向控制台泄漏。
    """
    if name:
        logger = logging.getLogger(f"{LOGGER_NAME}.{name}")
    else:
        logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers and not _configured:
        logger.addHandler(logging.NullHandler())
    return logger


def reset_for_tests():
    """测试辅助：拆掉全部 handler、复位幂等标记。"""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    for h in list(logger.handlers):
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)
    _configured = False
