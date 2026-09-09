"""全局声纹聚类回归测试：同一说话人的跨块本地标签必须塌缩为一个说话人，
不同说话人必须保持分离。

背景：旧的分块接缝标签拼接（_merge_chunk_labels）只在相邻块 30s 重叠区
贪心配对，同一人没在接缝同时出声就被当新人，12 块 × 3~4 人虚高到 42 个
说话人。_cluster_speakers_global 改为块内标签声纹原型 + 全局层次聚类。

本测试构造 3 个「真实说话人」（256 维随机基向量），每人在若干分块里以
不同本地标签（C0_SPEAKER_00 / C1_SPEAKER_02 / ...）出现，每段声纹 =
基向量 + 噪声（L2 归一）。聚类后必须恰好得到 3 人，且同一真实说话人的
所有本地标签映射到同一个全局说话人名。

不加载 ASR 模型、不打开音频设备、不发网络请求。
运行：.venv\\Scripts\\python.exe tests\\test_speaker_clustering.py
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import app  # noqa: E402

rng = np.random.default_rng(42)
DIM = 256

# 3 个真实说话人：256 维空间里的随机单位向量（跨说话人 cosine ≈ 0）
bases = []
for _ in range(3):
    v = rng.standard_normal(DIM).astype("float32")
    bases.append(v / np.linalg.norm(v))


def noisy(base, scale=0.05):
    # 256 维下 scale=0.05 噪声范数约 0.8，同人 cosine 约 0.78，
    # 与真实 wespeaker 同人相似度（0.7~0.9）一致。
    v = base + scale * rng.standard_normal(DIM).astype("float32")
    return (v / np.linalg.norm(v)).tolist()


# 12 个分块；每块 pyannote 产出 2~3 个本地标签，映射到真实说话人
tracks = []
t_cursor = 0.0
for ci in range(12):
    present = [s for s in range(3) if rng.random() > 0.15]
    if not present:
        present = [ci % 3]
    for li, spk in enumerate(present):
        label = f"C{ci}_SPEAKER_{li:02d}"
        for _ in range(rng.integers(1, 5)):
            dur = float(rng.uniform(1.5, 6.0))
            tracks.append({
                "start": t_cursor,
                "end": t_cursor + dur,
                "label": label,
                "emb": noisy(bases[spk]),
            })
            t_cursor += dur + 0.3

n_labels = len(set(t["label"] for t in tracks))
print(f"fabricated {len(tracks)} segments, {n_labels} local labels")

merged, cluster_embs = app.PyannoteDiarizer._cluster_speakers_global(
    tracks, num_speakers=None, status_callback=None)

spk_names = sorted(set(t[2] for t in merged))
print("clustered speakers:", spk_names)
assert len(spk_names) == 3, (
    f"expected 3 speakers, got {len(spk_names)}: {spk_names}")

# 每个本地标签的声纹原型（与 _cluster_speakers_global 同算法）
label_emb = defaultdict(list)
for t in tracks:
    label_emb[t["label"]].append(np.asarray(t["emb"], dtype="float32"))
protos = {}
for lab, em in label_emb.items():
    p = np.mean(np.stack(em), axis=0)
    protos[lab] = p / np.linalg.norm(p)

# 用 (start,end) 把输入段和合并结果对上，拿本地标签 -> 全局说话人名
seg_to_name = {(round(m[0], 3), round(m[1], 3)): m[2] for m in merged}
label_names = defaultdict(set)
for t in tracks:
    name = seg_to_name.get((round(t["start"], 3), round(t["end"], 3)))
    if name:
        label_names[t["label"]].add(name)

for lab, names in label_names.items():
    assert len(names) == 1, f"label {lab} split across {names}"

# 按最近真实基向量把本地标签分组，确认同组只对应一个全局说话人
base_labels = defaultdict(set)
for lab, p in protos.items():
    sims = [float(np.dot(p, b)) for b in bases]
    base_labels[int(np.argmax(sims))].add(lab)

ok = True
for spk, labs in base_labels.items():
    names = set()
    for lab in labs:
        names |= label_names[lab]
    print(f"real speaker {spk}: {len(labs)} local labels -> global {sorted(names)}")
    if len(names) != 1:
        ok = False

assert ok, "same real speaker split into multiple global speakers!"
print("\nPASS: 3 real speakers, cross-chunk labels correctly merged, no false splits.")
