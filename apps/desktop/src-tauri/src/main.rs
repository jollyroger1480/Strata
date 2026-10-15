//! Strata Desktop — Tauri shell.
//!
//! The shell is a *carrier*, not an owner. It does exactly three things:
//!
//!   1. finds its bundled resources (the private Node runtime, the App Host, the built UI) and
//!      starts the App Host on a free loopback port;
//!   2. opens one webview window on the host's own origin, so the renderer and the API share an
//!      origin exactly as the browser build does and the supplied UI's relative `/api/...` calls
//!      keep working with no rewrite;
//!   3. stops the host when the app exits, and sees to it that nothing the app started outlives it.
//!
//! One app instance runs at a time (the single-instance plugin): a second launch focuses the window that is
//! already open and starts nothing, so two apps never fight over the `app-N` engine slots. One app can still run
//! several engines (one per GPU, say): the host gives each its own `app-N` profile under Strata Control.
//!
//! Closing ends the engines this app started, on every path:
//!   - a normal exit and the window's X: the host is asked to stop them through Control (drained, then ended), with a
//!     deadline after which the host is ended;
//!   - the shell is killed, crashes, or the user logs off: on Windows the shell owns a kill-on-close Job Object that
//!     contains the host, Control, the owner host, the server and the engine, so they end with it; everywhere the host
//!     watches its parent and each engine's owner is told to follow the host (`--die-with-pid`).
//! Engines the Terminal started are not in any of this and are left alone.
//!
//! Ownership stays where the architecture puts it. The App Host owns the local client API and the
//! engine lifecycle; Strata Control owns components, models, profiles and jobs; DeepSeek Harness
//! owns the agent loop and durable sessions. There is deliberately no second supervisor, no
//! conversation store and no agent loop in this file.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

/// The kill-on-close Job Object that holds everything this app starts (Windows). Its only handle is this process's,
/// which the system closes however the process ends; closing the last handle ends every process in the job.
#[cfg(windows)]
mod job {
    use std::ffi::c_void;

    type Handle = *mut c_void;
    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x2000;
    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;

    #[repr(C)]
    #[derive(Default)]
    struct BasicLimits {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoCounters {
        counts: [u64; 6],
    }

    #[repr(C)]
    #[derive(Default)]
    struct ExtendedLimits {
        basic: BasicLimits,
        io: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(attributes: *mut c_void, name: *const u16) -> Handle;
        fn SetInformationJobObject(job: Handle, class: i32, info: *mut c_void, len: u32) -> i32;
        fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
        fn GetCurrentProcess() -> Handle;
        fn CloseHandle(handle: Handle) -> i32;
        fn IsProcessInJob(process: Handle, job: Handle, result: *mut i32) -> i32;
    }

    /// Is this process inside any job object? (The self-test asks, so an installed build proves it is contained.)
    pub fn current_process_in_a_job() -> bool {
        let mut result = 0i32;
        unsafe { IsProcessInJob(GetCurrentProcess(), std::ptr::null_mut(), &mut result) != 0 && result != 0 }
    }

    pub struct Job(Handle);

    // The handle is a plain kernel object handle, usable from any thread.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        /// A new job that ends all of its processes when its last handle closes. None if the system refuses.
        pub fn kill_on_close() -> Option<Job> {
            unsafe {
                let handle = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
                if handle.is_null() {
                    return None;
                }
                let mut info = ExtendedLimits::default();
                info.basic.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    handle,
                    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    &mut info as *mut ExtendedLimits as *mut c_void,
                    std::mem::size_of::<ExtendedLimits>() as u32,
                );
                if ok == 0 {
                    CloseHandle(handle);
                    return None;
                }
                Some(Job(handle))
            }
        }

        /// Put a process (by handle) in the job; everything it starts later joins too.
        pub fn assign_raw(&self, process: *mut c_void) -> bool {
            unsafe { AssignProcessToJobObject(self.0, process) != 0 }
        }

        /// Put this very process in the job, so the host and everything below it are contained.
        pub fn assign_current(&self) -> bool {
            self.assign_raw(unsafe { GetCurrentProcess() })
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

/// Held for the life of the process. The handle is deliberately never closed by us: the system closes it when the
/// process ends, by any route, and that is the moment the job ends what is left in it.
#[cfg(windows)]
static APP_JOB: std::sync::OnceLock<job::Job> = std::sync::OnceLock::new();

/// Is the app inside a job object right now? Always false off Windows.
fn app_is_contained() -> bool {
    #[cfg(windows)]
    {
        job::current_process_in_a_job()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// Put this process (and so everything it starts) in a kill-on-close job. True if it is contained now.
fn contain_app() -> bool {
    #[cfg(windows)]
    {
        if let Some(job) = job::Job::kill_on_close() {
            if job.assign_current() {
                let _ = APP_JOB.set(job);
                return true;
            }
        }
        eprintln!("[strata] could not put the app in a job object; engines are stopped by the host and their owners instead");
        false
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// The running App Host, kept so it can be stopped on exit.
struct HostProcess(Mutex<Option<Child>>);

/// The port the App Host bound, so the window (and the frontend) can be pointed at it.
struct HostPort(Mutex<Option<u16>>);

/// The secret that lets this shell (and nothing else) ask the host for a clean stop.
struct ShutdownToken(Mutex<Option<String>>);

/// 256 random bits as hex, from the standard library's per-process random hasher keys (no extra crate).
fn random_token() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    (0..4)
        .map(|i| {
            let mut h = RandomState::new().build_hasher();
            h.write_u32(i);
            h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
            format!("{:016x}", h.finish())
        })
        .collect()
}

/// The Control component that ships beside the host (`<resources>/control`): the bundled CPython, the locked
/// wheels, strata_control and the API server. None in a development checkout.
fn control_dir(res: &Path) -> Option<PathBuf> {
    let dir = res.join("control");
    let python = if cfg!(windows) { dir.join("python").join("python.exe") } else { dir.join("python").join("bin").join("python3") };
    if python.exists() && dir.join("strata.py").exists() {
        Some(dir)
    } else {
        None
    }
}

/// What every host start gets, packaged or self-test: acquiring is forbidden, Control is where the bundle put it.
fn apply_runtime_policy(cmd: &mut Command, res: &Path) {
    cmd.env("STRATA_NO_ACQUIRE", "1");
    if let Some(dir) = control_dir(res) {
        cmd.env("STRATA_CONTROL_DIR", dir);
    }
}

/// Ask the OS for an unused loopback port.
///
/// The port is released before the host binds it, so there is a small race; the host reports a
/// clear "port already in use" error if another process takes it first, and that surfaces in the
/// window instead of failing silently.
fn pick_port() -> Result<u16, String> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|e| format!("bind: {e}"))?;
    let port = listener.local_addr().map_err(|e| format!("addr: {e}"))?.port();
    drop(listener);
    Ok(port)
}

/// Locate a bundled resource.
///
/// In a packaged `.app` Tauri places `bundle.resources` under `Contents/Resources`. During
/// development they are not copied, so the repository layout is the fallback. Both are tried so
/// `tauri dev` and the packaged bundle behave the same way.
fn resource_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if let Ok(dir) = app.path().resource_dir() {
        if dir.join("host").join("server").join("index.mjs").exists() {
            return Ok(plain_path(dir));
        }
    }
    // development: apps/desktop/src-tauri -> ../../host  and  ./resources
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest)
}

/// Tauri returns Windows paths in the verbatim form (`\\?\C:\...`). Node cannot take such a path as
/// its entry script (`EISDIR ... lstat 'C:'`), so the host died at once and the window sat
/// unresponsive until the health wait gave up. Use the ordinary drive (or UNC) form.
fn plain_path(p: PathBuf) -> PathBuf {
    let text = p.to_string_lossy().into_owned();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    match text.strip_prefix(r"\\?\") {
        Some(rest) => PathBuf::from(rest),
        None => p,
    }
}

/// File name of the bundled Node runtime: `node.exe` on Windows, `node` elsewhere.
const NODE_EXE: &str = if cfg!(windows) { "node.exe" } else { "node" };

/// Where `tauri dev` finds a Node sidecar prepared next to the checkout (`toolchain/node`).
fn dev_node_candidate(res: &Path) -> PathBuf {
    let root = res.join("..").join("..").join("..").join("toolchain").join("node");
    if cfg!(windows) {
        // the Windows zip has node.exe at the top of the folder, not under bin/
        root.join("node-v24.12.0-win-x64").join("node.exe")
    } else {
        root.join("node-v24.12.0-darwin-arm64").join("bin").join("node")
    }
}

/// The private Node runtime that ships with the bundle.
fn node_binary(res: &Path) -> Result<PathBuf, String> {
    let bundled = res.join("node").join(NODE_EXE);
    if bundled.exists() {
        return Ok(bundled);
    }
    // development fallback: the sidecar prepared alongside the checkout
    let dev = dev_node_candidate(res);
    if dev.exists() {
        return Ok(dev);
    }
    Err(format!(
        "the bundled Node runtime is missing (looked for {} and {})",
        bundled.display(),
        dev.display()
    ))
}

/// A packaged GUI app must not flash a console window for the host process on Windows.
#[cfg(windows)]
fn hide_console(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_console(_cmd: &mut Command) {}

/// Wait until the host answers `/api/health`, or give up.
fn wait_for_health(
    port: u16,
    timeout: Duration,
    mut exited: impl FnMut() -> Option<String>,
) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    let url = format!("http://127.0.0.1:{port}/api/health");
    let mut last = String::from("no attempt made");
    while Instant::now() < deadline {
        // A host that already died will never answer: say so now instead of after the timeout.
        if let Some(why) = exited() {
            return Err(format!("the Strata App Host stopped before it became healthy: {why}"));
        }
        match ureq_get_ok(&url) {
            Ok(true) => return Ok(()),
            Ok(false) => last = "the host answered with a non-2xx status".into(),
            Err(e) => last = e,
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Err(format!("the Strata App Host did not become healthy at {url}: {last}"))
}

/// A minimal blocking HTTP POST with no body (the shell's one clean-stop request). Returns the status line.
fn http_post(url: &str, headers: &[(&str, &str)]) -> Result<String, String> {
    http_request("POST", url, headers).map(|(status, _)| status)
}

/// GET returning the response body (the self-test reads `/api/control/status`).
fn http_get_body(url: &str) -> Result<(String, String), String> {
    http_request("GET", url, &[])
}

fn http_request(method: &str, url: &str, headers: &[(&str, &str)]) -> Result<(String, String), String> {
    use std::io::{Read, Write};
    let rest = url.strip_prefix("http://").ok_or("only http:// is supported")?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let mut stream = std::net::TcpStream::connect_timeout(
        &authority.parse().map_err(|e| format!("bad authority {authority}: {e}"))?,
        Duration::from_millis(800),
    )
    .map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(20))).map_err(|e| e.to_string())?;
    let extra: String = headers.iter().map(|(k, v)| format!("{k}: {v}\r\n")).collect();
    stream
        .write_all(format!("{method} {path} HTTP/1.1\r\nHost: {authority}\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n").as_bytes())
        .map_err(|e| e.to_string())?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).map_err(|e| e.to_string())?;
    let status = buf.lines().next().unwrap_or_default().to_string();
    let body = buf.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
    Ok((status, body))
}

/// A minimal blocking HTTP GET, so the shell needs no async runtime for one health probe.
fn ureq_get_ok(url: &str) -> Result<bool, String> {
    use std::io::{Read, Write};
    let rest = url.strip_prefix("http://").ok_or("only http:// is supported")?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let mut stream = std::net::TcpStream::connect_timeout(
        &authority
            .parse()
            .map_err(|e| format!("bad authority {authority}: {e}"))?,
        Duration::from_millis(800),
    )
    .map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_millis(1500)))
        .map_err(|e| e.to_string())?;
    stream
        .write_all(format!("GET {path} HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\n\r\n").as_bytes())
        .map_err(|e| e.to_string())?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).map_err(|e| e.to_string())?;
    let status = buf.lines().next().unwrap_or_default();
    Ok(status.contains(" 200") || status.contains(" 204"))
}

/// Start the App Host and return the port it should serve on.
fn start_host(app: &tauri::AppHandle) -> Result<u16, String> {
    let res = resource_dir(app)?;
    let host_dir = if res.join("host").join("server").join("index.mjs").exists() {
        res.join("host")
    } else {
        res.join("..").join("host")
    };
    let entry = host_dir.join("server").join("index.mjs");
    if !entry.exists() {
        return Err(format!("the App Host entry is missing at {}", entry.display()));
    }
    let dist = host_dir.join("dist");
    if !dist.exists() {
        return Err(format!(
            "the built UI is missing at {}; run \"npm --prefix host run build\"",
            dist.display()
        ));
    }
    let node = node_binary(&res)?;

    let port = pick_port()?;

    let mut cmd = Command::new(&node);
    cmd.arg(&entry)
        .current_dir(&host_dir)
        .env("STRATA_PORT", port.to_string())
        .env("STRATA_DIST", &dist)
        .env("NODE_ENV", "production")
        // U12: a packaged build must have no success path that uses demo responses or demo
        // metrics. `STRATA_PACKAGED` makes the host refuse to simulate; `STRATA_REAL_ENGINE` is set
        // for the same reason, so the monitor reports `none` instead of fabricated numbers.
        .env("STRATA_PACKAGED", "1")
        .env("STRATA_REAL_ENGINE", "1")
        // the host leaves, after stopping what this app started, when this shell is gone (killed, crashed)
        .env("STRATA_EXIT_WITH_PARENT", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_runtime_policy(&mut cmd, &res);
    let token = random_token();
    cmd.env("STRATA_SHUTDOWN_TOKEN", &token);
    *app.state::<ShutdownToken>().0.lock().unwrap() = Some(token);
    hide_console(&mut cmd);
    // A user with the Strata API worker on a different port sets this; unset means the worker's
    // documented loopback default.
    if let Ok(url) = std::env::var("STRATA_SERVER_URL") {
        cmd.env("STRATA_SERVER_URL", url);
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("could not start the App Host on {}: {e}", node.display()))?;
    *app.state::<HostProcess>().0.lock().unwrap() = Some(child);
    *app.state::<HostPort>().0.lock().unwrap() = Some(port);

    let state_app = app.clone();
    wait_for_health(port, Duration::from_secs(60), move || {
        let state = state_app.state::<HostProcess>();
        let mut guard = state.0.lock().unwrap();
        match guard.as_mut().map(|c| c.try_wait()) {
            Some(Ok(Some(status))) => Some(format!("exit status {status}")),
            _ => None,
        }
    })?;
    Ok(port)
}

/// How long the host gets to stop the engines this app started (drain, then end) before it is ended. The host's own
/// deadline is shorter; whatever is still running when the shell exits is ended by the job object / the owners.
const HOST_STOP_SECONDS: u64 = 60;

/// Stop the App Host, and with it the engines this app started.
///
/// The host stops them through Control (profiles `app-N`), then exits. Engines another client started (the
/// Terminal's `strata runtime start`) are not touched. If the host does not finish in time it is ended, and the
/// engines are ended by the job object (Windows) or by their owners following the host (`--die-with-pid`).
fn stop_host(app: &tauri::AppHandle) {
    let port = app.try_state::<HostPort>().and_then(|s| *s.0.lock().unwrap());
    let token = app.try_state::<ShutdownToken>().and_then(|s| s.0.lock().unwrap().clone());
    if let Some(state) = app.try_state::<HostProcess>() {
        if let Some(mut child) = state.0.lock().unwrap().take() {
            // A clean stop first: the host stops, through Control, the engines this app started, then exits.
            // Engines another client started (the Terminal) are not touched. If it does not finish in time the
            // process is ended as before.
            if let (Some(port), Some(token)) = (port, token) {
                let _ = http_post(&format!("http://127.0.0.1:{port}/api/control/shutdown"), &[("x-strata-shutdown", token.as_str())]);
                let deadline = Instant::now() + Duration::from_secs(HOST_STOP_SECONDS);
                while Instant::now() < deadline {
                    if let Ok(Some(_)) = child.try_wait() {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Where a packaged build keeps `bundle.resources`, found from the executable's own path.
///
/// macOS `.app`: `Contents/MacOS/<exe>` -> `Contents/Resources`. Windows NSIS: next to the exe.
/// Linux deb/AppImage: `<prefix>/bin/<exe>` -> `<prefix>/lib/<productName>`.
fn packaged_resource_dir(exe: &Path) -> Option<PathBuf> {
    let dir = exe.parent()?;
    let mut candidates = vec![dir.to_path_buf()];
    if let Some(up) = dir.parent() {
        candidates.push(up.join("Resources"));
        candidates.push(up.join("lib").join("Strata"));
        candidates.push(up.join("lib").join("strata-desktop"));
    }
    candidates
        .into_iter()
        .find(|p| p.join("host").join("server").join("index.mjs").exists())
}

/// Parse `--self-test[=path]`; returns the report path when the flag is present.
fn self_test_path(args: &[String]) -> Option<String> {
    for a in args.iter().skip(1) {
        if a == "--self-test" {
            return Some(
                std::env::temp_dir()
                    .join("strata-desktop-self-test.json")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        if let Some(rest) = a.strip_prefix("--self-test=") {
            return Some(rest.to_string());
        }
    }
    None
}

/// Run `strata.py --version` on the bundle's own interpreter with a scrubbed environment.
fn control_runs(dir: &Path) -> (bool, String) {
    let python = if cfg!(windows) { dir.join("python").join("python.exe") } else { dir.join("python").join("bin").join("python3") };
    let mut cmd = Command::new(&python);
    cmd.arg(dir.join("strata.py")).arg("--version").current_dir(dir)
        .env_remove("PYTHONHOME").env_remove("PYTHONPATH").env_remove("VIRTUAL_ENV")
        .env("PYTHONNOUSERSITE", "1").env("PYTHONDONTWRITEBYTECODE", "1").env("STRATA_NO_ACQUIRE", "1");
    hide_console(&mut cmd);
    match cmd.output() {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let ok = out.status.success() && text.contains("python 3.12") && text.contains(&dir.display().to_string());
            (ok, if text.is_empty() { String::from_utf8_lossy(&out.stderr).trim().to_string() } else { text })
        }
        Err(e) => (false, format!("could not start {}: {e}", python.display())),
    }
}

/// With an empty component folder Control must say there is no engine and name the explicit command, exit 5,
/// and must not create or fetch anything.
fn control_engine_notice(dir: &Path) -> (bool, String) {
    let python = if cfg!(windows) { dir.join("python").join("python.exe") } else { dir.join("python").join("bin").join("python3") };
    let empty = std::env::temp_dir().join(format!("strata-self-test-components-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&empty);
    let mut cmd = Command::new(&python);
    cmd.arg(dir.join("strata.py")).args(["components", "engine", "--backend", "cuda", "--output", "json"])
        .arg("--components-dir").arg(&empty).current_dir(dir)
        .env_remove("PYTHONHOME").env_remove("PYTHONPATH")
        .env("PYTHONNOUSERSITE", "1").env("PYTHONDONTWRITEBYTECODE", "1").env("STRATA_NO_ACQUIRE", "1");
    hide_console(&mut cmd);
    let result = match cmd.output() {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout).to_string();
            let notice = text.contains("never downloads an engine");
            let created = empty.exists();
            (out.status.code() == Some(5) && notice && !created, format!("exit {:?}; notice={notice}; folder created={created}", out.status.code()))
        }
        Err(e) => (false, e.to_string()),
    };
    let _ = std::fs::remove_dir_all(&empty);
    result
}

/// Start the host from the packaged resources and report what actually happened.
fn run_self_test(report_path: &str) -> i32 {
    let mut steps: Vec<(String, bool, String)> = Vec::new();
    let mut record = |name: &str, ok: bool, detail: String| {
        steps.push((name.to_string(), ok, detail));
    };

    // Resolve resources the same way the real app does: `Contents/Resources` next to the
    // executable when packaged, the repository layout otherwise.
    let exe = std::env::current_exe().unwrap_or_default();
    let packaged_res = packaged_resource_dir(&exe);

    let res = match packaged_res {
        Some(p) => {
            record("resources-packaged", true, p.display().to_string());
            p
        }
        None => {
            let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            record(
                "resources-packaged",
                false,
                format!("not a packaged layout; using development root {}", dev.display()),
            );
            dev
        }
    };

    let host_dir = if res.join("host").join("server").join("index.mjs").exists() {
        res.join("host")
    } else {
        res.join("..").join("host")
    };
    let entry = host_dir.join("server").join("index.mjs");
    let dist = host_dir.join("dist");
    record("host-entry-present", entry.exists(), entry.display().to_string());
    record("ui-dist-present", dist.join("index.html").exists(), dist.display().to_string());

    let node = match node_binary(&res) {
        Ok(n) => {
            let mut probe = Command::new(&n);
            probe.arg("--version");
            hide_console(&mut probe);
            let version_ok = probe
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            record("private-node-runtime", version_ok, n.display().to_string());
            n
        }
        Err(e) => {
            record("private-node-runtime", false, e.clone());
            return write_self_test_report(report_path, steps, &e);
        }
    };

    // Closing must always end what the app started: on Windows the shell sits in a kill-on-close job object.
    if cfg!(windows) {
        record("shell-contained-in-kill-on-close-job", app_is_contained(), "IsProcessInJob".into());
    }

    let port = match pick_port() {
        Ok(p) => p,
        Err(e) => return write_self_test_report(report_path, steps, &e),
    };

    let mut cmd = Command::new(&node);
    cmd.arg(&entry)
        .current_dir(&host_dir)
        .env("STRATA_PORT", port.to_string())
        .env("STRATA_DIST", &dist)
        .env("STRATA_PACKAGED", "1")
        .env("STRATA_REAL_ENGINE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    apply_runtime_policy(&mut cmd, &res);
    hide_console(&mut cmd);
    let mut child = match cmd.spawn() {
        Ok(c) => {
            record("host-spawned", true, format!("pid {}", c.id()));
            c
        }
        Err(e) => {
            let msg = format!("spawn failed: {e}");
            record("host-spawned", false, msg.clone());
            return write_self_test_report(report_path, steps, &msg);
        }
    };

    let healthy = wait_for_health(port, Duration::from_secs(45), || match child.try_wait() {
        Ok(Some(status)) => Some(format!("exit status {status}")),
        _ => None,
    });
    record("host-healthy", healthy.is_ok(), match &healthy {
        Ok(()) => format!("http://127.0.0.1:{port}/api/health answered 2xx"),
        Err(e) => e.clone(),
    });

    // The UI must actually be served from the bundle, not merely present on disk.
    let ui_ok = ureq_get_ok(&format!("http://127.0.0.1:{port}/")).unwrap_or(false);
    record("ui-served", ui_ok, format!("http://127.0.0.1:{port}/"));

    // The API the renderer will call must be reachable on the same origin.
    let api_ok = ureq_get_ok(&format!("http://127.0.0.1:{port}/api/health")).unwrap_or(false);
    record("api-reachable-on-same-origin", api_ok, "GET /api/health".into());

    // The bundled Control component: present, runs on its own interpreter, and the host sees it with acquiring forbidden.
    match control_dir(&res) {
        Some(dir) => {
            record("control-component-present", true, dir.display().to_string());
            let (ok, detail) = control_runs(&dir);
            record("control-runs-on-bundled-python", ok, detail);
            let (ok, detail) = control_engine_notice(&dir);
            record("engine-missing-is-a-notice-not-a-download", ok, detail);
        }
        None => record("control-component-present", false, format!("no control component under {}", res.display())),
    }
    let status = http_get_body(&format!("http://127.0.0.1:{port}/api/control/status"));
    let (status_ok, status_detail) = match &status {
        Ok((line, body)) => {
            let parsed: serde_json::Value = serde_json::from_str(body).unwrap_or(serde_json::Value::Null);
            let forbidden = parsed["acquisition"]["forbidden"] == serde_json::Value::Bool(true);
            let control = parsed["control"]["python"].is_string();
            (line.contains(" 200") && forbidden && control, format!("{line}; acquisition.forbidden={forbidden}; control={control}"))
        }
        Err(e) => (false, e.clone()),
    };
    record("host-sees-control-and-forbids-acquiring", status_ok, status_detail);

    let _ = child.kill();
    let _ = child.wait();

    let msg = if steps.iter().all(|(_, ok, _)| *ok) {
        String::new()
    } else {
        "one or more steps failed".into()
    };
    write_self_test_report(report_path, steps, &msg)
}

/// Write the JSON report and return the process exit code.
fn write_self_test_report(path: &str, steps: Vec<(String, bool, String)>, error: &str) -> i32 {
    let ok = !steps.is_empty() && steps.iter().all(|(_, o, _)| *o);
    let steps_json: Vec<serde_json::Value> = steps
        .iter()
        .map(|(name, ok, detail)| {
            serde_json::json!({ "name": name, "ok": ok, "detail": detail })
        })
        .collect();
    let report = serde_json::json!({
        "schemaVersion": 1,
        "kind": "strata-desktop-self-test",
        "carrier": "tauri",
        "ok": ok,
        "error": error,
        "steps": steps_json,
    });
    let text = serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into());
    if let Err(e) = std::fs::write(path, &text) {
        eprintln!("[strata] could not write the self-test report to {path}: {e}");
    }
    println!("{text}");
    if ok {
        0
    } else {
        1
    }
}

fn main() {
    // ------------------------------------------------------------------ self-test
    //
    // `Strata.app/Contents/MacOS/strata-desktop --self-test[=path]` starts the App Host from the
    // *packaged resources*, waits for it to answer, checks that it serves the supplied UI, writes a
    // JSON report and exits 0/1. This is what makes the shipped artifact verifiable without a human
    // at the screen — the lesson from the Electron attempt, where a packaged app that never started
    // its host hung with no output at all.
    let args: Vec<String> = std::env::args().collect();
    contain_app();
    if let Some(path) = self_test_path(&args) {
        let code = run_self_test(&path);
        std::process::exit(code);
    }

    tauri::Builder::default()
        // First, as the plugin requires: a second launch ends here after telling the running app to show itself.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_shell::init())
        .manage(HostProcess(Mutex::new(None)))
        .manage(HostPort(Mutex::new(None)))
        .manage(ShutdownToken(Mutex::new(None)))
        .setup(|app| {
            let handle = app.handle().clone();
            let port = start_host(&handle).map_err(|e| {
                // A startup failure must be visible, not a blank window.
                eprintln!("[strata] {e}");
                Box::<dyn std::error::Error>::from(e)
            })?;

            let origin = format!("http://127.0.0.1:{port}");
            let url = origin.parse().expect("a valid loopback URL");

            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("Strata")
                .inner_size(1440.0, 920.0)
                .min_inner_size(900.0, 600.0)
                .build()?;

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                stop_host(&window.app_handle().clone());
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building the Strata application")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit = event {
                stop_host(app);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("strata-desktop-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn bundled_node_uses_the_platform_file_name() {
        let res = scratch("node");
        assert!(node_binary(&res).is_err());
        let dir = res.join("node");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(NODE_EXE), b"").unwrap();
        assert_eq!(node_binary(&res).unwrap(), dir.join(NODE_EXE));
        assert_eq!(NODE_EXE, if cfg!(windows) { "node.exe" } else { "node" });
        let _ = std::fs::remove_dir_all(&res);
    }

    #[test]
    fn verbatim_windows_paths_are_made_plain() {
        assert_eq!(plain_path(PathBuf::from(r"\\?\C:\a\b")), PathBuf::from(r"C:\a\b"));
        assert_eq!(
            plain_path(PathBuf::from(r"\\?\UNC\srv\share\x")),
            PathBuf::from(r"\\srv\share\x")
        );
        assert_eq!(plain_path(PathBuf::from("/usr/lib/Strata")), PathBuf::from("/usr/lib/Strata"));
    }

    #[test]
    fn control_is_found_only_when_whole() {
        let res = scratch("control");
        assert!(control_dir(&res).is_none());
        let dir = res.join("control");
        let py = if cfg!(windows) { dir.join("python").join("python.exe") } else { dir.join("python").join("bin").join("python3") };
        std::fs::create_dir_all(py.parent().unwrap()).unwrap();
        std::fs::write(&py, b"").unwrap();
        assert!(control_dir(&res).is_none(), "no strata.py yet");
        std::fs::write(dir.join("strata.py"), b"").unwrap();
        assert_eq!(control_dir(&res), Some(dir));
        let _ = std::fs::remove_dir_all(&res);
    }

    #[cfg(windows)]
    #[test]
    fn a_process_in_the_job_ends_when_the_job_is_closed() {
        use std::os::windows::io::AsRawHandle;
        let job = job::Job::kill_on_close().expect("a job object");
        let mut child = Command::new("ping").args(["-n", "120", "127.0.0.1"]).stdout(Stdio::null()).spawn().expect("ping");
        assert!(job.assign_raw(child.as_raw_handle() as *mut std::ffi::c_void), "the child joins the job");
        assert!(child.try_wait().unwrap().is_none(), "it runs while the job is open");
        drop(job); // what the system does when the app ends, by any route
        let deadline = Instant::now() + Duration::from_secs(10);
        while child.try_wait().unwrap().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let ended = child.try_wait().unwrap().is_some();
        if !ended {
            let _ = child.kill();
        }
        assert!(ended, "closing the job must end its processes");
    }

    #[test]
    fn the_shells_wait_for_the_host_exceeds_the_hosts_own_deadline_and_stays_bounded() {
        // the host ends itself after 45 s (server/index.mjs); the shell must wait longer than that, but not for ever
        assert!(HOST_STOP_SECONDS > 45 && HOST_STOP_SECONDS <= 90);
    }

    #[test]
    fn shutdown_tokens_are_long_and_different() {
        let (a, b) = (random_token(), random_token());
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn a_bare_node_is_not_taken_for_the_windows_runtime() {
        if !cfg!(windows) {
            return;
        }
        let res = scratch("bare");
        let dir = res.join("node");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("node"), b"").unwrap();
        assert!(node_binary(&res).is_err());
        let _ = std::fs::remove_dir_all(&res);
    }

    #[test]
    fn packaged_resources_are_found_for_each_platform_layout() {
        let root = scratch("layout");
        for (exe_rel, res_rel) in [
            ("Contents/MacOS/strata-desktop", "Contents/Resources"),
            ("install/strata-desktop.exe", "install"),
            ("usr/bin/strata-desktop", "usr/lib/Strata"),
        ] {
            let exe = root.join(exe_rel);
            let res = root.join(res_rel);
            assert!(packaged_resource_dir(&exe).is_none(), "{exe_rel}");
            let server = res.join("host").join("server");
            std::fs::create_dir_all(&server).unwrap();
            std::fs::write(server.join("index.mjs"), b"").unwrap();
            assert_eq!(packaged_resource_dir(&exe), Some(res), "{exe_rel}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
