"""Owner-only files on every platform.

POSIX keeps what the code always did (``0600`` files, ``0700`` directories). NTFS has no mode bits,
so on Windows the same promise is an ACL: a protected (non-inherited) DACL with exactly one
allow entry, for the current user. This uses ``advapi32`` through ``ctypes``; there is no
``pywin32`` or ``icacls`` dependency and no output parsing that depends on the display language.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

WINDOWS = os.name == "nt"

if WINDOWS:
    import ctypes
    from ctypes import wintypes

    _advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    _kernel.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel.LocalFree.restype = ctypes.c_void_p
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    _advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                            ctypes.POINTER(wintypes.DWORD)]
    _advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    _advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    _advapi.GetSecurityDescriptorDacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                                                  ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL)]
    _advapi.SetNamedSecurityInfoW.argtypes = [wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _advapi.SetNamedSecurityInfoW.restype = wintypes.DWORD
    _advapi.GetNamedSecurityInfoW.argtypes = [wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_void_p)]
    _advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
    _advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(wintypes.LPWSTR), ctypes.c_void_p]

_SE_FILE_OBJECT = 1
_DACL = 0x4
_PROTECTED_DACL = 0x80000000
_TOKEN_QUERY = 0x8
_TOKEN_USER = 1
_SDDL_REVISION_1 = 1
_sid_cache: str | None = None


def _winerr(what: str, code: int | None = None) -> OSError:
    return ctypes.WinError(code if code is not None else ctypes.get_last_error(), what)  # type: ignore[attr-defined]


def current_user_sid() -> str:
    """The string SID of the user this process runs as (Windows only)."""
    global _sid_cache
    if not WINDOWS:
        raise OSError("SIDs exist on Windows only")
    if _sid_cache:
        return _sid_cache
    token = wintypes.HANDLE()
    if not _advapi.OpenProcessToken(_kernel.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)):
        raise _winerr("OpenProcessToken")
    try:
        needed = wintypes.DWORD()
        _advapi.GetTokenInformation(token, _TOKEN_USER, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not _advapi.GetTokenInformation(token, _TOKEN_USER, buffer, needed, ctypes.byref(needed)):
            raise _winerr("GetTokenInformation")
        psid = ctypes.c_void_p.from_buffer(buffer).value  # TOKEN_USER starts with the SID pointer
        text = wintypes.LPWSTR()
        if not _advapi.ConvertSidToStringSidW(psid, ctypes.byref(text)):
            raise _winerr("ConvertSidToStringSidW")
        try:
            _sid_cache = text.value
        finally:
            _kernel.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    finally:
        _kernel.CloseHandle(token)
    assert _sid_cache
    return _sid_cache


def restrict_to_current_user(path: str | os.PathLike, *, directory: bool | None = None) -> None:
    """Make ``path`` readable and writable by the current user only.

    POSIX: ``chmod 0600`` (``0700`` for a directory). Windows: replace the DACL with one protected
    entry granting the current user full control (inherited by children when ``path`` is a
    directory) and drop every inherited entry.
    """
    path = os.fspath(path)
    if directory is None:
        directory = os.path.isdir(path)
    if not WINDOWS:
        os.chmod(path, 0o700 if directory else 0o600)
        return
    flags = "OICI" if directory else ""
    sddl = f"D:PAI(A;{flags};FA;;;{current_user_sid()})"
    descriptor = ctypes.c_void_p()
    if not _advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, _SDDL_REVISION_1, ctypes.byref(descriptor), None):
        raise _winerr("ConvertStringSecurityDescriptorToSecurityDescriptorW")
    try:
        present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
        if not _advapi.GetSecurityDescriptorDacl(descriptor, ctypes.byref(present), ctypes.byref(dacl),
                                                 ctypes.byref(defaulted)) or not present.value:
            raise _winerr("GetSecurityDescriptorDacl")
        code = _advapi.SetNamedSecurityInfoW(path, _SE_FILE_OBJECT, _DACL | _PROTECTED_DACL,
                                             None, None, dacl, None)
        if code:
            raise _winerr(f"SetNamedSecurityInfoW {path}", code)
    finally:
        _kernel.LocalFree(descriptor)


def dacl_sddl(path: str | os.PathLike) -> str:
    """The DACL of ``path`` as an SDDL string such as ``D:PAI(A;;FA;;;S-1-5-21-...)`` (Windows only)."""
    if not WINDOWS:
        raise OSError("DACLs exist on Windows only")
    descriptor = ctypes.c_void_p()
    code = _advapi.GetNamedSecurityInfoW(os.fspath(path), _SE_FILE_OBJECT, _DACL, None, None, None, None,
                                         ctypes.byref(descriptor))
    if code:
        raise _winerr(f"GetNamedSecurityInfoW {path}", code)
    try:
        text = wintypes.LPWSTR()
        if not _advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor, _SDDL_REVISION_1, _DACL, ctypes.byref(text), None):
            raise _winerr("ConvertSecurityDescriptorToStringSecurityDescriptorW")
        try:
            return text.value
        finally:
            _kernel.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    finally:
        _kernel.LocalFree(descriptor)


def is_private(path: str | os.PathLike) -> bool:
    """True when only the current user can reach ``path``.

    POSIX: no group or other permission bits. Windows: the DACL is protected (no inherited entries)
    and consists of exactly one allow entry for the current user.
    """
    if not WINDOWS:
        return not (os.lstat(path).st_mode & 0o077)
    sddl = dacl_sddl(path)
    head = sddl.split("(", 1)[0]
    entries = re.findall(r"\(([^()]*)\)", sddl)
    if not head.startswith("D:") or "P" not in head[2:] or len(entries) != 1:
        return False
    fields = entries[0].split(";")
    return len(fields) >= 6 and fields[0] == "A" and fields[5] == current_user_sid()


_PRIVILEGED = {"SY", "BA", "OW", "CO", "CG", "S-1-5-18", "S-1-5-32-544", "S-1-3-4"}


def is_unshared(path: str | os.PathLike) -> bool:
    """True when nobody but this user, SYSTEM and the administrators can reach ``path``.

    The relaxed form of :func:`is_private`, for files created under the user's profile where the
    inherited ACL already shuts out other accounts: any allow entry for Everyone, Users,
    Authenticated Users or another account makes it False. POSIX is the same check as
    :func:`is_private`.
    """
    if not WINDOWS:
        return is_private(path)
    sddl = dacl_sddl(path)
    allowed = _PRIVILEGED | {current_user_sid()}
    for entry in re.findall(r"\(([^()]*)\)", sddl):
        fields = entry.split(";")
        if len(fields) >= 6 and fields[0] in ("A", "OA") and fields[5] not in allowed:
            return False
    return True
