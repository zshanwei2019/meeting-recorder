# -*- coding: utf-8 -*-
"""音源下拉框「显示值 ↔ 配置值」一致性回归测试。

背景（实测踩坑）：#audioSource 的 <option> DOM 顺序是
  [0]=both(系统+麦克风) [1]=system(系统音频) [2]=mic(麦克风)
而 applyConfig 回填用的 srcMap 曾写成 {'system':0,'mic':1,'both':2}，
整体错位一位 → 配置存 system，界面却高亮「系统+麦克风」，用户被误导。

本测试直接解析 ui/index.html，断言 srcMap 的每个值都等于该 value
在 <option> 列表里的真实下标，不依赖浏览器、不加载 ASR。
运行：.venv\\Scripts\\python.exe tests\\test_audio_source_mapping.py
"""
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "ui" / "index.html"

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


src = io.open(HTML, encoding="utf-8").read()

# 1) 取出 #audioSource 这个 <select> 块内的 <option value=...>
sel_m = re.search(r'<select[^>]*id="audioSource"[^>]*>(.*?)</select>',
                  src, re.S)
check("能定位 #audioSource 下拉框", sel_m is not None)
if sel_m:
    options = re.findall(r'<option\s+value="([^"]+)"', sel_m.group(1))
    print("    选项 DOM 顺序:", options)
    check("有 3 个音源选项", len(options) == 3, f"got={options}")
    check("选项值集合 = both/system/mic",
          set(options) == {"both", "system", "mic"}, f"got={options}")

    # 2) 取出 srcMap 字面量
    map_m = re.search(r"const\s+srcMap\s*=\s*\{([^}]*)\}", src)
    check("能定位 srcMap", map_m is not None)
    if map_m:
        mapping = dict(re.findall(r"'([^']+)'\s*:\s*(\d+)", map_m.group(1)))
        mapping = {k: int(v) for k, v in mapping.items()}
        print("    srcMap:", mapping)
        check("srcMap 覆盖 both/system/mic",
              set(mapping) == {"both", "system", "mic"}, f"got={mapping}")
        # 3) 核心断言：映射下标必须等于该值在 DOM 中的真实位置
        for value, index in mapping.items():
            if value in options:
                check(f"srcMap['{value}']={index} 与选项实际位置一致",
                      options.index(value) == index,
                      f"实际位置={options.index(value)}")

# 4) 录音开始时发送的就是下拉框当前 .value（而不是别的来源）
check("录音开始读取 audioSource.value",
      "getElementById('audioSource').value" in src)

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
