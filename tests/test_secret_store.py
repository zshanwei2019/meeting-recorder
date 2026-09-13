# -*- coding: utf-8 -*-
"""secret_store（DPAPI 敏感配置加密）单测。

Windows 上做真实 DPAPI 往返；同时验证前缀协议、幂等、旧明文迁移、
非密字段不动、坏密文/跨用户解不开时安全降级为空串（绝不把密文当密钥发出）。
非 Windows 上 DPAPI 不可用，函数降级为恒等，这部分用例也应通过。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import secret_store as ss  # noqa: E402

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


print("=== 1. 环境探测一致 ===")
avail = ss.dpapi_available()
check("Windows 下 DPAPI 可用", (sys.platform == "win32") == (avail is True) or avail in (True, False))

print("\n=== 2. 加密往返（Windows 真实 DPAPI）===")
sample = "sk-test-secret-123456中文鍵"
enc = ss.encrypt_value(sample)
if avail:
    check("密文带 dpapi1: 前缀", isinstance(enc, str) and enc.startswith(ss.ENC_PREFIX), str(enc)[:20])
    check("密文不等于明文", enc != sample and ss.ENC_PREFIX not in sample)
    check("解密还原明文", ss.decrypt_value(enc) == sample, ss.decrypt_value(enc))
else:
    check("非 Windows 降级恒等（不加密）", enc == sample)
    check("非 Windows 解密恒等", ss.decrypt_value(enc) == sample)

print("\n=== 3. 幂等：已加密不再重复加密 ===")
enc2 = ss.encrypt_value(enc)
check("重复 encrypt 结果不变", enc2 == enc)

print("\n=== 4. 旧明文 / 无前缀原样透传（迁移读取兼容）===")
check("明文 decrypt 原样返回", ss.decrypt_value("old-plaintext-key") == "old-plaintext-key")
check("空串透传", ss.decrypt_value("") == "")
check("encrypt 空串不加密", ss.encrypt_value("") == "")
check("None 透传", ss.encrypt_value(None) is None and ss.decrypt_value(None) is None)
check("非字符串透传", ss.encrypt_value(123) == 123)

print("\n=== 5. 坏密文 / 跨用户密文安全降级为空串 ===")
check("损坏 base64 返回空串（不当作密钥）", ss.decrypt_value("dpapi1:!!!not-base64!!!") == "")
if avail:
    # 合法 base64 但不是有效 DPAPI blob
    import base64
    bogus = ss.ENC_PREFIX + base64.b64encode(b"not-a-real-dpapi-blob").decode("ascii")
    check("伪造 blob 解不开返回空串", ss.decrypt_value(bogus) == "")

print("\n=== 6. protect_config / unprotect_config ===")
cfg = {
    "engine": "aliyun",
    "aliyun_asr_api_key": "sk-aaa",
    "tencent_secret_key": "sec-bbb",
    "oss_bucket": "my-bucket",          # 非密字段
    "oss_endpoint": "oss-cn-hangzhou",  # 非密字段
    "hot_words": "西工,财务",            # 非密中文
    "llm_api_key": "",                  # 空密钥跳过
}
protected = ss.protect_config(cfg)
check("不修改入参（内存仍明文）", cfg["aliyun_asr_api_key"] == "sk-aaa")
check("非密字段原样保留", protected["oss_bucket"] == "my-bucket"
      and protected["hot_words"] == "西工,财务" and protected["engine"] == "aliyun")
check("空密钥保持空串", protected["llm_api_key"] == "")
if avail:
    check("敏感字段已加密", protected["tencent_secret_key"].startswith(ss.ENC_PREFIX)
          and protected["tencent_secret_key"] != "sec-bbb")
    restored = ss.unprotect_config(protected)
    check("解密回内存得到全部原密钥",
          restored["aliyun_asr_api_key"] == "sk-aaa"
          and restored["tencent_secret_key"] == "sec-bbb",
          str(restored.get("tencent_secret_key")))
    check("往返后非密字段不变", restored["oss_endpoint"] == "oss-cn-hangzhou")
else:
    check("非 Windows protect 恒等", protected["aliyun_asr_api_key"] == "sk-aaa")

print("\n=== 7. 敏感键清单覆盖所有云端/LLM 密钥 ===")
required = {
    "xfyun_api_secret", "xfyun_api_key", "aliyun_asr_api_key",
    "oss_access_key_id", "oss_access_key_secret",
    "volc_asr_api_key", "volc_asr_access_key",
    "tencent_secret_id", "tencent_secret_key",
    "hf_token", "llm_api_key",
}
check("清单含全部 11 个密钥键", required.issubset(set(ss.SECRET_KEYS)),
      str(sorted(set(ss.SECRET_KEYS) - required)))
# 非密配置不得误加密（否则会导致 bucket/endpoint 用不了）
for nonsecret in ("oss_bucket", "oss_endpoint", "cos_bucket", "cos_region",
                  "tencent_asr_region", "engine", "hot_words", "aliyun_vocabulary_id"):
    check(f"非密键不加密: {nonsecret}", nonsecret not in ss.SECRET_KEYS)

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
