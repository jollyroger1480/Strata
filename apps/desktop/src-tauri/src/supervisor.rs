//! Linux: the process that owns the host, and what it does when the app window cannot run.
//!
//! WebKitGTK can crash the process that shows the window (a null pointer deep inside libwebkit2gtk was seen on Ubuntu 26.04 with
//! NVIDIA on Wayland, and on Ubuntu 24.04 under Xvfb). When the window and the host lived in one process, that crash took the host
//! and the engine down with it. So on Linux the work is split in two processes of this one executable:
//!
//!   shell    (this module; no GTK, no WebKit)  starts the host, writes the session file, starts the window process, and watches both
//!   window   (`--window-child`; Tauri + WebKitGTK) shows the page, registers the app updater with the host, and nothing else
//!
//! If the window process dies for any reason except the person closing the window (a signal such as SIGSEGV, an abort, a failed
//! start, a missing library), the shell keeps the host running, tells the host to open the page in the default browser and to
//! show one notification, and prints the same sentence on the terminal. This is "browser mode"; `--browser` or STRATA_UI=browser
//! starts in it on purpose. In browser mode closing the tab does not stop the engine: the page has "Quit Strata", and `--quit`
//! ends a running Strata from the command line. The host listens on 127.0.0.1 only (the shell sets STRATA_BIND itself).
//!
//! The session file (`~/.strata/app-session.json`, mode 0600: process id, port, the secret only this user can read, the mode) is how
//! a second start finds the first one: it focuses the window (window mode) or opens another tab (browser mode) instead of
//! starting a second host.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

pub const BROWSER_NOTICE: &str = "Strata opened in your browser because the app window couldn't start on this computer.";

/// What the command line asked for.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// `--browser` or STRATA_UI=browser: no app window, the page opens in the default browser.
    pub browser: bool,
    /// `--quit`: end a running Strata and leave.
    pub quit: bool,
    /// `--window-child`: the window process (started by the shell, not by a person).
    pub window_child: bool,
    pub help: bool,
}

pub fn parse_args(args: &[String], ui_env: Option<&str>) -> Options {
    let mut o = Options { browser: matches!(ui_env.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("browser")), ..Options::default() };
    for a in args.iter().skip(1) {
        match a.as_str() {
            "--browser" => o.browser = true,
            "--quit" => o.quit = true,
            "--window-child" => o.window_child = true,
            "--help" | "-h" => o.help = true,
            _ => {}
        }
    }
    o
}

pub const HELP: &str = "Strata\n\
  (no option)  open the Strata app window\n\
  --browser    open Strata in your default browser instead of the app window (same as STRATA_UI=browser)\n\
  --quit       stop a running Strata (its engine too) and leave\n\
Closing the browser tab does not stop Strata in browser mode: use Quit Strata in the menu, or --quit.\n";

/// The command a person types to run this app again, for the messages ("run: Strata.AppImage --quit").
pub fn command_name() -> String {
    if let Ok(image) = std::env::var("APPIMAGE") {
        if let Some(name) = std::path::Path::new(&image).file_name() {
            return format!("./{}", name.to_string_lossy());
        }
    }
    std::env::current_exe().ok().and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).unwrap_or_else(|| "strata-desktop".into())
}

pub fn browser_chosen_line(port: u16) -> String {
    format!(
        "Strata is open in your browser at http://127.0.0.1:{port}/. Closing the tab does not stop Strata: choose Quit Strata in the menu, or run {} --quit.",
        command_name()
    )
}

pub fn open_by_hand_line(port: u16) -> String {
    format!("No browser could be opened. Open http://127.0.0.1:{port}/ in your browser.")
}

// ------------------------------------------------------------------------------------------------ the session file

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    pub pid: u32,
    pub port: u16,
    pub token: String,
    pub mode: String,
}

fn session_path() -> PathBuf {
    if let Some(p) = std::env::var_os("STRATA_APP_SESSION_FILE") {
        return PathBuf::from(p);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    home.join(".strata").join("app-session.json")
}

pub fn write_session(s: &Session) {
    let path = session_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let text = serde_json::json!({ "pid": s.pid, "port": s.port, "token": s.token, "mode": s.mode }).to_string();
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(&path) {
        Ok(mut f) => {
            let _ = f.write_all(text.as_bytes());
        }
        Err(e) => eprintln!("[strata] could not write {}: {e}", path.display()),
    }
}

pub fn read_session() -> Option<Session> {
    let text = std::fs::read_to_string(session_path()).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    Some(Session {
        pid: v["pid"].as_u64()? as u32,
        port: v["port"].as_u64()? as u16,
        token: v["token"].as_str()?.to_string(),
        mode: v["mode"].as_str().unwrap_or("window").to_string(),
    })
}

pub fn remove_session(only_if_pid: u32) {
    if read_session().map(|s| s.pid == only_if_pid).unwrap_or(false) {
        let _ = std::fs::remove_file(session_path());
    }
}

fn pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        true
    }
}

/// A Strata that is really running: its shell process exists and its host answers.
pub fn live_session() -> Option<Session> {
    let s = read_session()?;
    if s.pid != std::process::id() && pid_alive(s.pid) && crate::ureq_get_ok(&format!("http://127.0.0.1:{}/api/health", s.port)).unwrap_or(false) {
        Some(s)
    } else {
        None
    }
}

// ------------------------------------------------------------------------------------------------ talking to the host

/// POST to the host as the shell (the secret in the header). Returns the response body.
fn host_post(port: u16, token: &str, path: &str, body: &serde_json::Value) -> Result<String, String> {
    let (status, text) = crate::http_request_body("POST", &format!("http://127.0.0.1:{port}{path}"), &[("x-strata-shutdown", token), ("content-type", "application/json")], Some(&body.to_string()))?;
    if status.contains(" 200") || status.contains(" 202") {
        Ok(text)
    } else {
        Err(format!("{status}: {text}"))
    }
}

/// Did the host manage to open the page (the "opened" flag in its answer)?
fn opened(answer: &str) -> bool {
    let body = answer.rsplit("\r\n\r\n").next().unwrap_or(answer);
    serde_json::from_str::<serde_json::Value>(body.trim()).ok().and_then(|v| v["opened"].as_bool()).unwrap_or(false)
}

/// Tell the host: everyone uses a browser tab now (and open it, and show `notice`). Prints what happened.
pub fn switch_host_to_browser(port: u16, token: &str, reason: &str, notice: &str) {
    let body = serde_json::json!({ "mode": "browser", "reason": reason, "notice": notice, "open": true });
    match host_post(port, token, "/api/control/ui", &body) {
        Ok(answer) => {
            if !opened(&answer) {
                eprintln!("[strata] {}", open_by_hand_line(port));
            }
        }
        Err(e) => eprintln!("[strata] could not tell the host to open your browser ({e}). Open http://127.0.0.1:{port}/ in your browser."),
    }
}

/// Open the page in the default browser without changing the host's mode (a second start while Strata runs).
fn open_page_only(port: u16, token: &str) {
    match host_post(port, token, "/api/control/ui", &serde_json::json!({ "open": true })) {
        Ok(answer) if opened(&answer) => {}
        _ => eprintln!("[strata] {}", open_by_hand_line(port)),
    }
}

pub fn describe_exit(status: &ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            let name = match sig {
                6 => " (abort)",
                11 => " (segmentation fault)",
                7 => " (bus error)",
                9 => " (killed)",
                15 => " (terminated)",
                _ => "",
            };
            return format!("signal {sig}{name}");
        }
    }
    match status.code() {
        Some(c) => format!("exit code {c}"),
        None => "ended without a code".into(),
    }
}

// ------------------------------------------------------------------------------------------------ the shell

fn spawn_window(port: u16, token: &str) -> std::io::Result<Child> {
    let exe = std::env::current_exe()?;
    Command::new(exe)
        .arg("--window-child")
        .env("STRATA_WINDOW_PORT", port.to_string())
        .env("STRATA_SHUTDOWN_TOKEN", token)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
}

/// `--quit`: end a running Strata (the window, the host, the engines it started) and wait for it.
pub fn quit_running() -> i32 {
    let Some(s) = live_session() else {
        println!("Strata is not running.");
        let _ = std::fs::remove_file(session_path());
        return 0;
    };
    match host_post(s.port, &s.token, "/api/control/shutdown", &serde_json::json!({})) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("[strata] could not ask Strata to stop ({e}).");
            return 1;
        }
    }
    let deadline = Instant::now() + Duration::from_secs(crate::HOST_STOP_SECONDS + 10);
    while Instant::now() < deadline {
        if !pid_alive(s.pid) {
            println!("Strata has stopped.");
            return 0;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    eprintln!("[strata] Strata is still stopping its engine. Try again in a moment.");
    1
}

/// A second start while Strata already runs: focus the window, or open another tab. Starts no second host.
fn join_running(s: &Session, browser: bool) -> i32 {
    if browser || s.mode == "browser" {
        println!("Strata is already running. Opening it in your browser.");
        open_page_only(s.port, &s.token);
        return 0;
    }
    // window mode: the window process's single-instance check brings the open window forward and exits at once
    match spawn_window(s.port, &s.token).and_then(|mut c| c.wait()) {
        Ok(status) if status.success() => 0,
        Ok(status) => {
            eprintln!("[strata] The app window stopped ({}). Opening Strata in your browser.", describe_exit(&status));
            open_page_only(s.port, &s.token);
            0
        }
        Err(e) => {
            eprintln!("[strata] could not start the app window ({e}). Opening Strata in your browser.");
            open_page_only(s.port, &s.token);
            0
        }
    }
}

/// The shell. Returns the process exit code.
pub fn run(opts: &Options) -> i32 {
    if opts.help {
        print!("{HELP}");
        return 0;
    }
    if opts.quit {
        return quit_running();
    }
    if let Some(s) = live_session() {
        return join_running(&s, opts.browser);
    }

    let exe = std::env::current_exe().unwrap_or_default();
    let res = crate::packaged_resource_dir(&exe).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let port = match crate::pick_stable_port() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[strata] {e}");
            return 1;
        }
    };
    let token = crate::random_token();
    let mode = if opts.browser { "browser" } else { "window" };
    let mut cmd = match crate::host_command(&res, port, &token) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[strata] {e}");
            return 1;
        }
    };
    cmd.env("STRATA_UI_MODE", mode).env("STRATA_APP_VERSION", env!("CARGO_PKG_VERSION"));
    if opts.browser {
        cmd.env("STRATA_UI_REASON", "chosen");
    }
    let mut host = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[strata] could not start the App Host: {e}");
            return 1;
        }
    };
    let health = crate::wait_for_health(port, Duration::from_secs(60), || match host.try_wait() {
        Ok(Some(status)) => Some(format!("exit status {status}")),
        _ => None,
    });
    if let Err(e) = health {
        eprintln!("[strata] {e}");
        let _ = host.kill();
        let _ = host.wait();
        return 1;
    }
    let me = std::process::id();
    write_session(&Session { pid: me, port, token: token.clone(), mode: mode.into() });

    let mut in_browser = opts.browser;
    let mut window: Option<Child> = None;
    if in_browser {
        eprintln!("[strata] {}", browser_chosen_line(port));
        switch_host_to_browser(port, &token, "chosen", "");
    } else {
        match spawn_window(port, &token) {
            Ok(c) => window = Some(c),
            Err(e) => {
                eprintln!("[strata] could not start the app window ({e}).");
                eprintln!("[strata] {BROWSER_NOTICE}");
                in_browser = true;
                switch_host_to_browser(port, &token, "window-failed", BROWSER_NOTICE);
                write_session(&Session { pid: me, port, token: token.clone(), mode: "browser".into() });
            }
        }
    }

    let code = loop {
        std::thread::sleep(Duration::from_millis(200));
        // The host ended (Quit Strata, a crash, someone killed it): nothing is left to show.
        if let Ok(Some(status)) = host.try_wait() {
            if let Some(w) = window.as_mut() {
                let _ = w.kill();
                let _ = w.wait();
            }
            break if status.success() { 0 } else { eprintln!("[strata] The App Host stopped ({}).", describe_exit(&status)); 1 };
        }
        let Some(w) = window.as_mut() else { continue };
        let Ok(Some(status)) = w.try_wait() else { continue };
        window = None;
        if status.success() {
            // the person closed the window: stop the host and the engines this app started, as before
            crate::stop_host_child(&mut host, Some(port), Some(&token));
            break 0;
        }
        // The window stopped without being closed: keep Strata running and move to the browser.
        eprintln!("[strata] The app window stopped unexpectedly ({}).", describe_exit(&status));
        eprintln!("[strata] {BROWSER_NOTICE}");
        in_browser = true;
        switch_host_to_browser(port, &token, "window-failed", BROWSER_NOTICE);
        write_session(&Session { pid: me, port, token: token.clone(), mode: "browser".into() });
    };
    let _ = in_browser;
    remove_session(me);
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        std::iter::once("strata-desktop").chain(list.iter().copied()).map(String::from).collect()
    }

    #[test]
    fn the_command_line_picks_browser_quit_and_the_window_child() {
        assert_eq!(parse_args(&args(&[]), None), Options::default());
        assert!(parse_args(&args(&["--browser"]), None).browser);
        assert!(parse_args(&args(&[]), Some("browser")).browser);
        assert!(parse_args(&args(&[]), Some(" Browser ")).browser);
        assert!(!parse_args(&args(&[]), Some("window")).browser);
        assert!(parse_args(&args(&["--quit"]), None).quit);
        assert!(parse_args(&args(&["--window-child"]), None).window_child);
        assert!(parse_args(&args(&["-h"]), None).help);
        // anything else (a file the desktop hands over, an AppImage option) is ignored
        assert_eq!(parse_args(&args(&["--appimage-extract-and-run", "x.gguf"]), None), Options::default());
    }

    #[test]
    fn the_browser_message_is_the_plain_sentence_and_names_the_way_out() {
        assert_eq!(BROWSER_NOTICE, "Strata opened in your browser because the app window couldn't start on this computer.");
        let line = browser_chosen_line(47831);
        assert!(line.contains("http://127.0.0.1:47831/") && line.contains("--quit") && line.contains("Quit Strata"));
        assert!(open_by_hand_line(47831).contains("127.0.0.1:47831"));
    }

    #[test]
    fn a_session_round_trips_and_is_private() {
        let dir = std::env::temp_dir().join(format!("strata-session-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("STRATA_APP_SESSION_FILE", dir.join("s.json"));
        assert!(read_session().is_none());
        let s = Session { pid: 4242, port: 47831, token: "ab".repeat(32), mode: "browser".into() };
        write_session(&s);
        assert_eq!(read_session(), Some(s.clone()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(dir.join("s.json")).unwrap().permissions().mode() & 0o777, 0o600);
        }
        // only the owner removes it
        remove_session(1);
        assert!(read_session().is_some());
        remove_session(4242);
        assert!(read_session().is_none());
        std::env::remove_var("STRATA_APP_SESSION_FILE");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_hosts_answer_tells_whether_a_browser_opened() {
        assert!(opened("HTTP/1.1 200 OK\r\n\r\n{\"mode\":\"browser\",\"opened\":true}"));
        assert!(!opened("HTTP/1.1 200 OK\r\n\r\n{\"mode\":\"browser\",\"opened\":false}"));
        assert!(!opened("not json"));
    }
}
