"""`strata connect <harness>`: the setup text for an app that talks to a running Strata server.

Each harness is one entry of `HARNESSES` below: what it speaks, where its config file lives, the text to put there, the
environment variables, and whether that config has been **tested** (phaseA-tests/split/harness-configs/<harness>.md, section 4 of
TUI-LAUNCH-SPEC.md: one prompt with the answer checked and one tool call with its arguments checked, at 32K).

TODO table (an entry says `tested=False` until its tested config replaces the draft below; the draft comes from the app's public
documentation and is labelled "not tested yet" on screen):

    opencode  codex  claude-code  omp  pi  hermes  openclaw   -> tested configs from the section-4 worker
    deepseek-harness                                          -> built into the Strata app, nothing to write

Placeholders in the templates: {url} (server root, no /v1), {key}, {model}, {port}.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileSpec:
    path: str                      # may start with ~ ; {home} {appdata} are not needed, ~ is expanded
    kind: str                      # json | toml | yaml | text | env
    content: str                   # template
    merge: bool = False            # json only: merge into an existing file instead of replacing it


@dataclass
class Harness:
    id: str
    title: str
    speaks: str                    # "chat + tools", "responses + tools", ...
    files: list[FileSpec] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    tested: bool = False
    built_in: bool = False
    aliases: tuple[str, ...] = ()
    install: str = ""

    @property
    def status(self) -> str:
        return "built into the Strata app" if self.built_in else ("tested" if self.tested else "not tested yet")


def _j(obj) -> str:
    return json.dumps(obj, indent=2)


# --- drafts: replaced by the tested files of harness-configs/ (see the TODO table above) --------------------------------------
HARNESSES: list[Harness] = [
    Harness("opencode", "OpenCode", "chat + tools",
            files=[FileSpec("~/.config/opencode/opencode.json", "json", _j({
                "$schema": "https://opencode.ai/config.json",
                "provider": {"strata": {"npm": "@ai-sdk/openai-compatible", "name": "Strata",
                                        "options": {"baseURL": "{url}/v1", "apiKey": "{key}"},
                                        "models": {"{model}": {"name": "Strata ({model})"}}}},
                "model": "strata/{model}"}), merge=True)],
            install="npm i -g opencode-ai"),
    Harness("codex", "Codex CLI", "responses + tools",
            files=[FileSpec("~/.codex/config.toml", "toml",
                            'model = "{model}"\nmodel_provider = "strata"\n\n[model_providers.strata]\nname = "Strata"\n'
                            'base_url = "{url}/v1"\nenv_key = "STRATA_API_KEY"\nwire_api = "responses"\n')],
            env={"STRATA_API_KEY": "{key}"}, install="npm i -g @openai/codex"),
    Harness("claude-code", "Claude Code", "messages + tools",
            files=[FileSpec("~/.claude/settings.json", "json", _j({"env": {
                "ANTHROPIC_BASE_URL": "{url}", "ANTHROPIC_AUTH_TOKEN": "{key}", "ANTHROPIC_MODEL": "{model}",
                "ANTHROPIC_SMALL_FAST_MODEL": "{model}", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}}), merge=True)],
            env={"ANTHROPIC_BASE_URL": "{url}", "ANTHROPIC_AUTH_TOKEN": "{key}", "ANTHROPIC_MODEL": "{model}"},
            notes=["No Anthropic login is needed: the server answers with your local model."],
            aliases=("claude",), install="npm i -g @anthropic-ai/claude-code"),
    Harness("omp", "oh-my-pi (omp)", "chat + tools",
            files=[FileSpec("~/.omp/agent/models.yml", "yaml",
                            "providers:\n  strata:\n    baseUrl: {url}/v1\n    apiKey: {key}\n    api: openai-completions\n"
                            "    models:\n      - id: {model}\n        name: Strata ({model})\n")],
            aliases=("oh-my-pi",)),
    Harness("pi", "pi (pi.dev)", "chat + tools",
            files=[FileSpec("~/.pi/agent/models.json", "json", _j({"providers": {"strata": {
                "baseUrl": "{url}/v1", "api": "openai-completions", "apiKey": "{key}",
                "models": [{"id": "{model}"}]}}}), merge=True)]),
    Harness("hermes", "Hermes Agent", "chat + tools",
            files=[FileSpec("~/.hermes/config.yaml", "yaml",
                            "model:\n  provider: custom\n  base_url: {url}/v1\n  api_key: {key}\n  default: {model}\n")],
            aliases=("hermes-agent",)),
    Harness("openclaw", "OpenClaw", "chat + tools",
            files=[FileSpec("~/.openclaw/openclaw.json", "json", _j({"models": {"mode": "merge", "providers": {"strata": {
                "baseUrl": "{url}/v1", "apiKey": "{key}", "api": "openai-completions",
                "models": [{"id": "{model}", "name": "Strata ({model})"}]}}},
                "agents": {"defaults": {"model": {"primary": "strata/{model}"}}}}), merge=True)]),
    Harness("deepseek-harness", "DeepSeek Harness", "the Strata app", built_in=True,
            notes=["Strata's own coding harness lives in the Strata app: open the app, choose Code, and it uses this server."],
            aliases=("deepseek",)),
]

# The tested configs, when they exist, override the drafts: harness-configs/<id>.json beside this file (made from the .md files).
OVERRIDE_DIR = Path(__file__).with_name("harness_configs")


def load_overrides() -> None:
    if not OVERRIDE_DIR.is_dir():
        return
    by_id = {h.id: h for h in HARNESSES}
    for path in sorted(OVERRIDE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            h = by_id.get(data["id"]) or Harness(data["id"], data.get("title", data["id"]), data.get("speaks", ""))
            h.title = data.get("title", h.title)
            h.speaks = data.get("speaks", h.speaks)
            h.files = [FileSpec(**f) for f in data.get("files", [])]
            h.env = dict(data.get("env", {}))
            h.notes = list(data.get("notes", []))
            h.tested = bool(data.get("tested", True))
            h.install = data.get("install", h.install)
            if h.id not in by_id:
                HARNESSES.append(h)
        except (OSError, ValueError, KeyError, TypeError):
            continue


load_overrides()


def find(name: str) -> Harness | None:
    key = name.strip().lower()
    for h in HARNESSES:
        if key == h.id or key in h.aliases or key == h.title.lower():
            return h
    return None


def closest_name(name: str) -> str | None:
    from strata_cli.launch.settings import closest
    return closest(name, [h.id for h in HARNESSES])


# ------------------------------------------------------------------------------------------------ rendering

def fill(text: str, *, url: str, key: str, model: str, port: int) -> str:
    """Replace the placeholders; the key goes in only where the caller passed a real one."""
    out = text
    for k, v in (("{url}", url.rstrip("/")), ("{key}", key), ("{model}", model), ("{port}", str(port))):
        out = out.replace(k, v)
    return out


def render(h: Harness, *, url: str, key: str, model: str) -> dict:
    """What to show or write for this harness: files with their text, and environment lines."""
    m = re.search(r":(\d+)(?:/|$)", url)
    port = int(m.group(1)) if m else (443 if url.startswith("https") else 80)
    files = []
    for f in h.files:
        files.append({"path": f.path, "kind": f.kind, "merge": f.merge,
                      "text": fill(f.content, url=url, key=key, model=model, port=port)})
    env = {k: fill(v, url=url, key=key, model=model, port=port) for k, v in h.env.items()}
    return {"harness": h.id, "title": h.title, "speaks": h.speaks, "status": h.status, "files": files, "env": env,
            "notes": list(h.notes), "install": h.install}


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


def write_file(spec: dict, *, backup: bool = True) -> dict:
    """Write one rendered file. An existing file is copied to `<name>.strata-bak-<time>` first; a JSON file with `merge` keeps the
    keys it already had (the new ones win on the same key). Returns {path, backup, action}."""
    path = expand(spec["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.is_file()
    bak = None
    text = spec["text"]
    if existed and backup:
        bak = path.with_name(path.name + ".strata-bak-" + time.strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(path, bak)
    if existed and spec.get("merge") and spec["kind"] == "json":
        try:
            old = json.loads(path.read_text(encoding="utf-8-sig"))
            new = json.loads(text)
            text = json.dumps(_merge(old, new), indent=2)
        except ValueError:
            pass                                    # an old file that is not JSON is replaced (a backup was made)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)                       # it can hold the API key
    except OSError:
        pass
    return {"path": str(path), "backup": str(bak) if bak else None, "action": "merged" if existed and spec.get("merge") else
            ("replaced" if existed else "created")}


def _merge(old, new):
    if isinstance(old, dict) and isinstance(new, dict):
        out = dict(old)
        for k, v in new.items():
            out[k] = _merge(old[k], v) if k in old else v
        return out
    return new


def mask(text: str, key: str) -> str:
    return text.replace(key, key[:10] + "•" * 8) if key else text
