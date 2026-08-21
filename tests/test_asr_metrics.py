"""ASR 指标函数回归测试。

复用 tools/asr_benchmark.py 内置的 selftest()，避免与工具里的断言重复维护。
这些指标零第三方依赖，因此在任何 venv 下都能跑（不需要 faster-whisper/funasr）。

运行：.venv\\Scripts\\python.exe tests\\test_asr_metrics.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import asr_benchmark  # noqa: E402


def main():
    rc = asr_benchmark.selftest()

    # 额外确认：指标模块不得在 import 期拉入 ASR 依赖，
    # 否则 CI / 无模型环境就跑不了指标测试了。
    leaked = [m for m in ("faster_whisper", "ctranslate2", "funasr", "torch")
              if m in sys.modules]
    if leaked:
        print("FAIL 指标模块意外导入了 ASR 依赖: {}".format(", ".join(leaked)))
        return 1
    print("OK   指标模块未拉入任何 ASR 依赖（可在无模型环境运行）")
    return rc


if __name__ == "__main__":
    sys.exit(main())
