# -*- coding: utf-8 -*-
"""模型加载锁 + 错误透出回归测试。

背景：PR #13 切换文件转写默认模型后，点「实时转写」报
"FunASR模型加载失败"。根因有二：
1. 启动预加载与用户点击实时转写并发加载多个大模型（含同一份
   1.1GB 标点模型被两个线程重复构造），可能撞 modelscope 文件锁/
   内存峰值；
2. 底层回调已带具体异常，但调用方用泛化句 "FunASR模型加载失败"
   覆盖，导致用户和排查者都看不到真正原因。

本测试不加载真模型，用桩验证：
- 三个 load_*_model 共用一把锁，串行化；
- load_stream_model 复用已加载的 punc_model，不重复构造；
- 加载失败时具体错误能被调用方拿到（不再被泛化句吞掉）。
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

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


print("=== 1. 锁存在且三个加载方法共用同一把 ===")
t = app.FunASRTranscriber()
check("有 _model_load_lock", hasattr(t, "_model_load_lock"))
check("是 threading.Lock", isinstance(t._model_load_lock, type(threading.Lock())))

# 用桩替换 AutoModel，记录每次构造是否在锁内、以及调用时序
in_critical = {"file": False, "stream": False, "diar": False}
order = []
lock_held_during = {"file": None, "stream": None, "diar": None}


class FakeModel:
    def __init__(self, tag):
        self.tag = tag
        # 构造期间检测：全局锁是否被持有
        lock_held_during[tag] = t._model_load_lock.acquire(blocking=False) is False
        if lock_held_during[tag] is False:
            # 没拿到说明锁没被持有（acquire 成功），立即释放
            t._model_load_lock.release()
        order.append(tag)
        time.sleep(0.05)  # 放大并发窗口


def fake_automodel(*args, **kwargs):
    # 根据 model= 关键字判断是哪个模型
    m = kwargs.get("model", "")
    if "online" in m:
        return FakeModel("stream")
    if "spk_model" in kwargs or m.endswith("8404-pytorch"):
        return FakeModel("diar")
    if "punc" in m:
        return FakeModel("punc")
    return FakeModel("file")


# 用 monkeypatch 替换 funasr.AutoModel
import types
fake_funasr = types.ModuleType("funasr")
fake_funasr.AutoModel = fake_automodel
sys.modules["funasr"] = fake_funasr

print("\n=== 2. 三个加载方法构造模型时都持有锁 ===")
t.load_file_model(model_name="iic/test-file")
t.load_stream_model(model_name="iic/test-online")
t2 = app.FunASRTranscriber()
# 重置，单独测 diar
in_critical
t2._model_load_lock = t._model_load_lock  # 共用锁便于观测
t2._load_diarization_model()

check("file 模型在锁内构造", lock_held_during["file"] is True, lock_held_during)
check("stream 模型在锁内构造", lock_held_during["stream"] is True, lock_held_during)
check("diar 模型在锁内构造", lock_held_during["diar"] is True, lock_held_during)

print("\n=== 3. 并发加载被串行化（无重叠构造）===")
# 重新用一个干净实例，记录时间区间
t3 = app.FunASRTranscriber()
intervals = []
orig_fake = fake_funasr.AutoModel


def fake_automodel2(*args, **kwargs):
    start = time.time()
    time.sleep(0.15)
    end = time.time()
    intervals.append((start, end, kwargs.get("model", "?")))
    return object()


fake_funasr.AutoModel = fake_automodel2

th1 = threading.Thread(target=lambda: t3.load_file_model(model_name="iic/a"))
th2 = threading.Thread(target=lambda: t3.load_stream_model(model_name="iic/online-a"))
th1.start()
time.sleep(0.02)
th2.start()
th1.join(timeout=30)
th2.join(timeout=30)

check("至少两个模型被构造", len(intervals) >= 2, len(intervals))
# 检查区间不重叠
overlap = False
for i in range(len(intervals)):
    for j in range(i + 1, len(intervals)):
        s1, e1, _ = intervals[i]
        s2, e2, _ = intervals[j]
        if s1 < e2 and s2 < e1:
            overlap = True
check("构造区间互不重叠（串行化）", not overlap, intervals)

print("\n=== 4. load_stream_model 复用已加载的 punc_model ===")
fake_funasr.AutoModel = orig_fake
t4 = app.FunASRTranscriber()
sentinel_punc = object()
t4.punc_model = sentinel_punc  # 模拟文件转写已加载标点模型
constructed = []


def fake_automodel3(*args, **kwargs):
    constructed.append(kwargs.get("model", ""))
    return object()


fake_funasr.AutoModel = fake_automodel3
t4.load_stream_model(model_name="iic/online-x")
check("stream 模型被构造", any("online-x" in c for c in constructed))
check("没有重复构造标点模型", not any("punc" in c for c in constructed), constructed)
check("punc_model 保持为已加载实例", t4.punc_model is sentinel_punc)

print("\n=== 5. 加载失败时具体错误透出（不被泛化句吞掉）===")
t5 = app.FunASRTranscriber()


def boom(*args, **kwargs):
    raise RuntimeError("磁盘满了 / 权重损坏 XYZ")


fake_funasr.AutoModel = boom
got = {}
ok = t5.load_stream_model(
    model_name="iic/online-fail",
    status_callback=lambda s, m: got.setdefault("msgs", []).append((s, m)),
)
check("加载返回 False", ok is False)
err_msgs = [m for s, m in got.get("msgs", []) if s == "error"]
check("有 error 回调", len(err_msgs) > 0)
check("错误消息含具体原因",
      any("磁盘满了" in m for m in err_msgs), err_msgs)
check("错误消息不是泛化句",
      not any(m.strip() == "FunASR模型加载失败" for m in err_msgs), err_msgs)

# 清理 sys.modules，避免污染其它测试
del sys.modules["funasr"]

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
