//! Strata Desktop — Tauri shell.
//!
//! The shell is a *carrier*, not an owner. It does exactly three things:
//!
//!   1. finds its bundled resources (the private Node runtime, the App Host, the built UI) and
//!      starts the App Host on a free loopback port;
//!   2. opens one webview window on the host's own origin, so the renderer and the API share an
//!      origin exactly as the browser build does and the supplied UI's relative `/api/...` calls
//!      keep working with no rewrite;
//!   3. stops the host when the app exits.
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

/// The running App Host, kept so it can be stopped on exit.
struct HostProcess(Mutex<Option<Child>>);

/// The port the App Host bound, so the window (and the frontend) can be pointed at it.
struct HostPort(Mutex<Option<u16>>);

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
            return Ok(dir);
        }
    }
    // development: apps/desktop/src-tauri -> ../../host  and  ./resources
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest)
}

/// The private Node runtime that ships with the bundle.
fn node_binary(res: &Path) -> Result<PathBuf, String> {
    let bundled = res.join("node").join("node");
    if bundled.exists() {
        return Ok(bundled);
    }
    // development fallback: the sidecar prepared alongside the checkout
    let dev = res
        .join("..")
        .join("..")
        .join("..")
        .join("toolchain")
        .join("node")
        .join("node-v24.12.0-darwin-arm64")
        .join("bin")
        .join("node");
    if dev.exists() {
        return Ok(dev);
    }
    Err(format!(
        "the bundled Node runtime is missing (looked for {} and {})",
        bundled.display(),
        dev.display()
    ))
}

/// Wait until the host answers `/api/health`, or give up.
fn wait_for_health(port: u16, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    let url = format!("http://127.0.0.1:{port}/api/health");
    let mut last = String::from("no attempt made");
    while Instant::now() < deadline {
        match ureq_get_ok(&url) {
            Ok(true) => return Ok(()),
            Ok(false) => last = "the host answered with a non-2xx status".into(),
            Err(e) => last = e,
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Err(format!("the Strata App Host did not become healthy at {url}: {last}"))
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
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
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

    wait_for_health(port, Duration::from_secs(60))?;
    Ok(port)
}

/// Stop the App Host.
///
/// Only this process is signalled. The engine worker and any jobs belong to Strata Control and are
/// deliberately not touched here: quitting the window must not kill inference another client is
/// using.
fn stop_host(app: &tauri::AppHandle) {
    if let Some(state) = app.try_state::<HostProcess>() {
        if let Some(mut child) = state.0.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
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

/// Start the host from the packaged resources and report what actually happened.
fn run_self_test(report_path: &str) -> i32 {
    let mut steps: Vec<(String, bool, String)> = Vec::new();
    let mut record = |name: &str, ok: bool, detail: String| {
        steps.push((name.to_string(), ok, detail));
    };

    // Resolve resources the same way the real app does: `Contents/Resources` next to the
    // executable when packaged, the repository layout otherwise.
    let exe = std::env::current_exe().unwrap_or_default();
    let packaged_res = exe
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("Resources"))
        .filter(|p| p.join("host").join("server").join("index.mjs").exists());

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
            let version_ok = Command::new(&n)
                .arg("--version")
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

    let healthy = wait_for_health(port, Duration::from_secs(45));
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
    if let Some(path) = self_test_path(&args) {
        let code = run_self_test(&path);
        std::process::exit(code);
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(HostProcess(Mutex::new(None)))
        .manage(HostPort(Mutex::new(None)))
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
