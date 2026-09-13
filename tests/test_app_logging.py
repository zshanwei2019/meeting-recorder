# -*- coding: utf-8 -*-
"""统一日志模块 app_logging 的单测：文件轮转、幂等、无控制台环境兜底。"""
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app_logging as al  # noqa: E402

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  OK   {name}")
        passed += 1
    else:
        print(f"  FAIL {name} {detail}")
        failed += 1


tmpdir = tempfile.mkdtemp()

# 每个用例前复位，保证互不干扰
al.reset_for_tests()

print("=== 1. 未初始化时 get_logger 安全（NullHandler，不打印不报错）===")
lg0 = al.get_logger("standalone")
check("未配置时挂了 NullHandler",
      any(isinstance(h, logging.NullHandler) for h in lg0.handlers))
lg0.info("这条不应出现在控制台（被 NullHandler 吞掉）")
check("logger 名带统一前缀", lg0.name == "meeting_recorder.standalone", lg0.name)

print("\n=== 2. setup_logging 写文件（轮转 handler）===")
logger = al.setup_logging(log_dir=tmpdir, console=False)
logger.info("云端转写开始 provider=tencent")
logger.error("转写失败 err=鉴权失败")
for h in logger.handlers:
    h.flush()
log_path = Path(tmpdir) / al.DEFAULT_LOG_FILE
check("日志文件已创建", log_path.exists())
content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
check("info 落盘", "云端转写开始" in content)
check("error 落盘", "转写失败" in content)
check("带级别与时间格式", "[ERROR]" in content and "meeting_recorder" in content)

print("\n=== 3. 幂等：重复 setup 不重复挂 handler ===")
n_before = len(logger.handlers)
al.setup_logging(log_dir=tmpdir, console=False)
n_after = len(logging.getLogger(al.LOGGER_NAME).handlers)
check("第二次 setup handler 数不变", n_after == n_before, f"{n_before} -> {n_after}")

print("\n=== 4. 子 logger 冒泡到统一文件 ===")
child = al.get_logger("cloud_asr")
child.warning("火山极速版 429 将重试")
for h in logging.getLogger(al.LOGGER_NAME).handlers:
    h.flush()
content2 = log_path.read_text(encoding="utf-8")
check("子 logger 日志进入同一文件", "429" in content2 and "cloud_asr" in content2)

print("\n=== 5. 轮转参数正确 ===")
rfh = next((h for h in logging.getLogger(al.LOGGER_NAME).handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)), None) if False else None
import logging.handlers
rfh = next((h for h in logging.getLogger(al.LOGGER_NAME).handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)), None)
check("存在 RotatingFileHandler", rfh is not None)
if rfh:
    check("单文件 5MB", rfh.maxBytes == 5 * 1024 * 1024, rfh.maxBytes)
    check("保留 5 份备份", rfh.backupCount == 5, rfh.backupCount)
    check("UTF-8 编码（中文不乱码）", rfh.encoding and rfh.encoding.lower().replace("-", "") == "utf8")

print("\n=== 6. propagate 关闭，不向 root 冒泡重复 ===")
check("主 logger propagate=False", logging.getLogger(al.LOGGER_NAME).propagate is False)

print("\n=== 7. 文件目录不可用时不崩溃（降级）===")
al.reset_for_tests()
bad_path = str(Path(tmpdir) / "not_a_dir_but_a_file")
Path(bad_path).write_text("x", encoding="utf-8")  # 占成普通文件，mkdir 失败
lg_bad = al.setup_logging(log_dir=bad_path, console=False, file_log=True)
check("目录异常时仍返回 logger，不抛异常", lg_bad is not None)

print("\n=== 8. reset_for_tests 还原 ===")
al.reset_for_tests()
check("reset 后 handler 清空",
      len(logging.getLogger(al.LOGGER_NAME).handlers) == 0)

# 复位到“未配置 + NullHandler”状态，避免污染同进程后续（各测试文件独立进程，双保险）
al.get_logger()

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
