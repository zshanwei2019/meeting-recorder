# -*- coding: utf-8 -*-
"""敏感配置字段的本地加密存储（Windows DPAPI）。

为什么：各家云端 ASR / OSS / LLM 的密钥此前以明文存在 ~/MeetingRecorder/config.json。
其中 OSS/COS 的 AK/SK 权限远大于普通 API Key，机器被他人读到文件或同步到网盘即泄露。

方案：用 Windows 自带 DPAPI（CryptProtectData）加密。密文绑定当前 Windows 用户，
同一机器换用户/换机器都解不开，且无需用户管理额外主密钥。加密结果加 ``dpapi1:``
前缀后 base64，仍可安全地放进 JSON。

- 旧明文配置：读取时原样保留（不认识前缀即视为明文），下次保存自动迁移成密文。
- 非 Windows（开发/Mac）或 DPAPI 不可用：降级为明文并由调用方记日志，不拖垮主程序。
- 纯标准库（ctypes）；只有真正加解密时才加载 crypt32。
"""
from __future__ import annotations

import base64
import sys

# 需要加密的配置键（API Key / Secret / Token；app_id、bucket、endpoint 等非密项不加密）
SECRET_KEYS = (
    "xfyun_api_secret",
    "xfyun_api_key",
    "aliyun_asr_api_key",
    "oss_access_key_id",
    "oss_access_key_secret",
    "volc_asr_api_key",
    "volc_asr_access_key",
    "tencent_secret_id",
    "tencent_secret_key",
    "hf_token",
    "llm_api_key",
)

ENC_PREFIX = "dpapi1:"


def dpapi_available() -> bool:
    """仅 Windows 且能加载 crypt32 时可用。"""
    if sys.platform != "win32":
        return False
    try:
        import ctypes  # noqa: F401
        return bool(getattr(ctypes, "windll", None) and ctypes.windll.crypt32)
    except Exception:
        return False


def encrypt_value(plaintext: str) -> str:
    """DPAPI 加密一个字符串，返回 dpapi1:<base64>；非 Windows 降级返回原文。"""
    if plaintext is None:
        return plaintext
    if not isinstance(plaintext, str):
        return plaintext
    # 空串不加密（保护配置会跳过空密钥，密文里也不应出现空值的 blob）
    if not plaintext:
        return plaintext
    # 已加密的不重复加密（幂等）
    if plaintext.startswith(ENC_PREFIX):
        return plaintext
    if not dpapi_available():
        return plaintext

    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    data = plaintext.encode("utf-8")
    buf = ctypes.create_string_buffer(data, len(data))
    in_blob = DATA_BLOB(
        len(data),
        ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)),
    )
    out_blob = DATA_BLOB()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptProtectData(
        ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)
    ):
        raise OSError("CryptProtectData 失败（Windows DPAPI）")
    try:
        raw = ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)
    return ENC_PREFIX + base64.b64encode(raw).decode("ascii")


def decrypt_value(stored: str) -> str:
    """解密 dpapi1: 密文；不带前缀的（旧明文/非 Windows 降级）原样返回。"""
    if not isinstance(stored, str) or not stored.startswith(ENC_PREFIX):
        return stored
    if not dpapi_available():
        # 密文却没有 DPAPI（跨平台拷贝了配置）——无法解，返回空避免把密文当密钥发出去
        return ""

    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    try:
        raw = base64.b64decode(stored[len(ENC_PREFIX):].encode("ascii"))
    except Exception:
        return ""
    buf = ctypes.create_string_buffer(raw, len(raw))
    in_blob = DATA_BLOB(
        len(raw),
        ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)),
    )
    out_blob = DATA_BLOB()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptUnprotectData(
        ctypes.byref(in_blob), None, None, None, None, 0,
        ctypes.byref(out_blob),
    ):
        # 当前用户解不开（换机/换账号）：返回空，由上层提示重新填写
        return ""
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData).decode("utf-8")
    finally:
        kernel32.LocalFree(out_blob.pbData)


def protect_config(config: dict) -> dict:
    """返回副本：敏感字段加密后再落盘。不修改入参；空值跳过。"""
    out = dict(config)
    for k in SECRET_KEYS:
        v = out.get(k)
        if isinstance(v, str) and v:
            out[k] = encrypt_value(v)
    return out


def unprotect_config(config: dict) -> dict:
    """返回副本：敏感字段解密到内存供业务使用。解不开的密文得到空串。"""
    out = dict(config)
    for k in SECRET_KEYS:
        v = out.get(k)
        if isinstance(v, str) and v:
            out[k] = decrypt_value(v)
    return out
