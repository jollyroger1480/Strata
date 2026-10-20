// strata/platform/thread_name.hpp - names the engine's long-lived threads, so that a person (or a script) can tell the
// pool workers from the host thread and the helpers when looking at where they run: Linux `ps -L -o tid,psr,comm` and
// /proc/<pid>/task/<tid>/comm, Windows GetThreadDescription (Process Explorer, WinDbg, the tools/ probes).
// A name changes nothing the thread does.  Header-only: the callers sit in several libraries that do not link each other.
#pragma once

#include <cstdio>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

namespace strata::platform {

/// Gives the calling thread a name: "strata-pool-3" for `name_this_thread("strata-pool", 3)`, "strata-host" for no index.
/// Linux keeps 15 characters, the rest is cut.  Never fails loudly: a system that cannot name a thread leaves it unnamed.
inline void name_this_thread(const char* base, int index = -1) {
    char name[64];
    if (index >= 0) std::snprintf(name, sizeof name, "%s-%d", base, index);
    else std::snprintf(name, sizeof name, "%s", base);
#if defined(_WIN32)
    // SetThreadDescription (Windows 10 1607 and later) is looked up, not linked: older SDKs and toolchains lack the import
    using SetDesc = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    static const SetDesc fn = (SetDesc) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription");
    if (fn == nullptr) return;
    wchar_t wide[64];
    int i = 0;
    for (; name[i] != '\0' && i < 63; ++i) wide[i] = (wchar_t) (unsigned char) name[i];
    wide[i] = L'\0';
    (void) fn(GetCurrentThread(), wide);
#elif defined(__linux__)
    name[15] = '\0';
    (void) pthread_setname_np(pthread_self(), name);
#elif defined(__APPLE__)
    (void) pthread_setname_np(name);
#endif
}

}  // namespace strata::platform
