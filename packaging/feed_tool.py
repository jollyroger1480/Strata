#!/usr/bin/env python3
"""The release-side tool for the update feed: keys, signed packages, archives, the signed feed.

Nothing here runs on a user's PC. It runs on the owner's release machine, with the owner's private key, and writes files
that are then uploaded (to a local folder or a local HTTP server while testing; to the real host only when the owner
decides). Nothing is uploaded by this tool.

Typical use (the owner's offline machine; replace ``KEYDIR`` with a folder that is NOT in a git checkout):

    python packaging/feed_tool.py keygen --out KEYDIR                      # once: feed-key.json (SECRET) + feed-key.pub.json
    python packaging/feed_tool.py sign-package <components>/engine/0.1.46-linux-x86_64-cuda --key KEYDIR/feed-key.json
    python packaging/feed_tool.py pack-engine  <that folder> --out RELEASE/engine           # archive + <name>.entry.json
    python packaging/feed_tool.py pack-control <control bundle folder> --version 0.1.46 --platform linux-x86_64 --out RELEASE/control
    python packaging/feed_tool.py build-feed --out RELEASE --sequence 12 --key KEYDIR/feed-key.json \
        --engine RELEASE/engine --control RELEASE/control [--app-json tauri-latest.json] [--channel stable] [--expires-days 365]
    python packaging/feed_tool.py verify-feed RELEASE --pub KEYDIR/feed-key.pub.json

``keygen`` never prints the secret and refuses to write inside a git checkout or over an existing file. The public file's
entry goes into ``strata_control/update_keys.json`` (``keys``) once, in the commit that ships the first signed release.
The Tauri updater key is a different key (minisign, made by ``tauri signer generate``); its public half goes in
``tauri.conf.json`` and in ``update_keys.json`` as ``app_minisign_public``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata_control import engine_component as ec, feed as feedmod, signing  # noqa: E402


def _inside_git(path: Path) -> bool:
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / ".git").exists():
            return True
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_key(path: Path) -> dict:
    key = json.loads(path.read_text(encoding="utf-8"))
    if key.get("alg") != signing.ALG or "seed" not in key:
        raise SystemExit(f"{path} is not a secret key file made by `feed_tool.py keygen`")
    return key


def keygen(out: Path, *, allow_git: bool = False) -> dict:
    out = Path(out)
    if _inside_git(out) and not allow_git:
        raise SystemExit(f"{out} is inside a git checkout; a private key must not be where it can be committed. Choose another folder.")
    secret, public = out / "feed-key.json", out / "feed-key.pub.json"
    if secret.exists() or public.exists():
        raise SystemExit(f"{secret} or {public} already exists; refusing to overwrite a key")
    out.mkdir(parents=True, exist_ok=True)
    key = signing.generate_key()
    fd = os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(key, stream, indent=1)
        stream.write("\n")
    from strata_control import private_files
    private_files.restrict_to_current_user(secret, directory=False)          # 0600, or a protected ACL on Windows
    public.write_text(json.dumps({"keys": [signing.public_part(key)]}, indent=1) + "\n", encoding="utf-8")
    return signing.public_part(key)


def sign_package(folder: Path, key: dict) -> str:
    """Put the signature into the package's manifest. Returns the manifest's SHA-256 afterwards (the feed names it)."""
    path = Path(folder) / ec.MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    manifest["signature"] = None
    payload = ec.signing_payload(manifest)
    manifest["signature"] = signing.sign_bytes(key, payload)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    ec.read_manifest(Path(folder))                      # still a valid manifest
    signing.verify_bytes({key["key_id"]: __import__("base64").b64decode(key["public"])}, payload, manifest["signature"])
    return _sha256(path)


def _mode(path: Path, rel: str, entrypoint: str | None) -> int:
    if rel == entrypoint:
        return 0o755
    try:
        return 0o755 if path.stat().st_mode & stat.S_IXUSR and os.name != "nt" else 0o644
    except OSError:
        return 0o644


def _write_archive(files: list[tuple[str, Path]], archive: Path, fmt: str, entrypoint: str | None) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "zip":
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel, path in files:
                zf.write(path, rel)
        return
    mode = {"tar.gz": "w:gz", "tar.xz": "w:xz"}[fmt]
    with tarfile.open(archive, mode) as tar:
        for rel, path in files:
            info = tar.gettarinfo(str(path), arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if info.issym():                              # a Control bundle's Python has relative symlinks; engines have none
                tar.addfile(info)
                continue
            info.mode = _mode(path, rel, entrypoint)
            with path.open("rb") as stream:
                tar.addfile(info, stream)


def _tree(folder: Path, *, links: bool = False) -> list[tuple[str, Path]]:
    """Regular files under `folder`; with `links` also the symbolic links, which must be relative and stay inside it."""
    found = []
    root = folder.resolve()
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            if not links:
                continue
            target = os.readlink(path)
            if os.path.isabs(target) or not (path.parent / target).resolve().is_relative_to(root):
                raise SystemExit(f"{path} is a symbolic link that leaves the package ({target}); a package may only link inside itself")
            found.append((path.relative_to(folder).as_posix(), path))
        elif path.is_file():
            found.append((path.relative_to(folder).as_posix(), path))
    return found


def pack_engine(folder: Path, out: Path, fmt: str = "tar.gz", *, url_prefix: str = "engine/") -> dict:
    """Archive a (signed) engine package folder and return its feed entry. The archive holds the folder's files at its root."""
    folder = Path(folder)
    component = ec.read_manifest(folder)
    files = _tree(folder)
    installed = sum(p.stat().st_size for _, p in files)
    name = f"strata-engine-{folder.name}.{fmt}"
    archive = Path(out) / name
    _write_archive(files, archive, fmt, component.entrypoint)
    entry = {"version": component.version, "platform": component.platform, "backend": component.backend,
             "variant": component.variant, "archs": list(component.archs), "min_driver": dict(component.min_driver),
             "min_runtime": component.min_runtime, "build_id": component.build_id, "format": fmt,
             "url": f"{url_prefix}{name}", "size": archive.stat().st_size, "sha256": _sha256(archive),
             "manifest_sha256": _sha256(folder / ec.MANIFEST), "installed_bytes": installed, "notes": component.notes}
    (Path(out) / (name + ".entry.json")).write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
    return entry


def pack_control(folder: Path, out: Path, *, version: str, platform: str, fmt: str = "tar.gz",
                 entry_file: str = "strata.py", url_prefix: str = "control/") -> dict:
    folder = Path(folder)
    files = _tree(folder, links=True)
    installed = sum(p.stat().st_size for _, p in files if not p.is_symlink())
    name = f"strata-control-{version}-{platform}.{fmt}"
    archive = Path(out) / name
    _write_archive(files, archive, fmt, entry_file)
    entry = {"platform": platform, "format": fmt, "url": f"{url_prefix}{name}", "size": archive.stat().st_size,
             "sha256": _sha256(archive), "installed_bytes": installed, "entry": entry_file, "version": version}
    (Path(out) / (name + ".entry.json")).write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
    return entry


def build_feed(out: Path, *, key: dict, sequence: int, channel: str = "stable", engine_dir: Path | None = None,
               control_dir: Path | None = None, app_json: Path | None = None, expires_days: int | None = None,
               now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    document: dict = {"schema_version": feedmod.SCHEMA_VERSION, "feed": "strata-updates", "channel": channel,
                      "sequence": sequence, "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "expires_at": (now + dt.timedelta(days=expires_days)).strftime("%Y-%m-%dT%H:%M:%SZ") if expires_days else None}
    if engine_dir:
        document["engine"] = {"packages": [json.loads(p.read_text(encoding="utf-8"))
                                           for p in sorted(Path(engine_dir).glob("*.entry.json"))]}
    if control_dir:
        entries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(control_dir).glob("*.entry.json"))]
        versions = {e["version"] for e in entries}
        if len(versions) > 1:
            raise SystemExit(f"the control packages are for different versions: {sorted(versions)}")
        if entries:
            document["control"] = {"version": versions.pop(), "packages": [{k: v for k, v in e.items() if k != "version"} for e in entries]}
    if app_json:
        document["app"] = json.loads(Path(app_json).read_text(encoding="utf-8"))
    payload = (json.dumps(document, indent=1, sort_keys=True) + "\n").encode("utf-8")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / feedmod.FEED_NAME).write_bytes(payload)
    signature = signing.sign_bytes(key, payload)
    (Path(out) / (feedmod.FEED_NAME + feedmod.SIGNATURE_SUFFIX)).write_text(json.dumps(signature) + "\n", encoding="utf-8")
    if "app" in document:                       # the file the Tauri updater plugin reads
        latest = Path(out) / "app" / "latest.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps({k: v for k, v in document["app"].items()}, indent=1) + "\n", encoding="utf-8")
    # a parse of what was written is the last check: a feed this tool makes must be a feed Control accepts
    feedmod.verify_and_parse(payload, json.dumps(signature).encode(), source=str(Path(out) / feedmod.FEED_NAME),
                             trusted={key["key_id"]: __import__("base64").b64decode(key["public"])}, now=now)
    return document


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("--out", type=Path, required=True)
    sp = sub.add_parser("sign-package")
    sp.add_argument("folder", type=Path)
    sp.add_argument("--key", type=Path, required=True)
    pe = sub.add_parser("pack-engine")
    pe.add_argument("folder", type=Path)
    pe.add_argument("--out", type=Path, required=True)
    pe.add_argument("--format", choices=feedmod.FORMATS, default="tar.gz")
    pe.add_argument("--url-prefix", default="engine/", help='where the archive sits relative to the feed ("" for flat GitHub release assets)')
    pc = sub.add_parser("pack-control")
    pc.add_argument("folder", type=Path)
    pc.add_argument("--out", type=Path, required=True)
    pc.add_argument("--version", required=True)
    pc.add_argument("--platform", required=True, choices=feedmod.PLATFORMS)
    pc.add_argument("--format", choices=feedmod.FORMATS, default="tar.gz")
    pc.add_argument("--url-prefix", default="control/")
    bf = sub.add_parser("build-feed")
    bf.add_argument("--out", type=Path, required=True)
    bf.add_argument("--key", type=Path, required=True)
    bf.add_argument("--sequence", type=int, required=True)
    bf.add_argument("--channel", default="stable")
    bf.add_argument("--engine", type=Path)
    bf.add_argument("--control", type=Path)
    bf.add_argument("--app-json", type=Path)
    bf.add_argument("--expires-days", type=int)
    vf = sub.add_parser("verify-feed")
    vf.add_argument("source")
    vf.add_argument("--pub", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "keygen":
        public = keygen(a.out)
        print(f"wrote {a.out / 'feed-key.json'} (SECRET, keep it offline and backed up) and {a.out / 'feed-key.pub.json'}")
        print(f"key_id {public['key_id']}")
    elif a.cmd == "sign-package":
        print("manifest sha256", sign_package(a.folder, _load_key(a.key)))
    elif a.cmd == "pack-engine":
        entry = pack_engine(a.folder, a.out, a.format, url_prefix=a.url_prefix)
        print(json.dumps({k: entry[k] for k in ("version", "backend", "variant", "size", "sha256", "manifest_sha256")}, indent=1))
    elif a.cmd == "pack-control":
        entry = pack_control(a.folder, a.out, version=a.version, platform=a.platform, fmt=a.format, url_prefix=a.url_prefix)
        print(json.dumps({k: entry[k] for k in ("version", "platform", "size", "sha256")}, indent=1))
    elif a.cmd == "build-feed":
        doc = build_feed(a.out, key=_load_key(a.key), sequence=a.sequence, channel=a.channel, engine_dir=a.engine,
                         control_dir=a.control, app_json=a.app_json, expires_days=a.expires_days)
        print(f"wrote {a.out / feedmod.FEED_NAME} (+ .sig); engines {len((doc.get('engine') or {}).get('packages', []))}, "
              f"control {'yes' if 'control' in doc else 'no'}, app {'yes' if 'app' in doc else 'no'}")
    elif a.cmd == "verify-feed":
        trusted = signing.load_trusted_keys(a.pub, environ={})
        loaded = feedmod.fetch(a.source, trusted)
        print(f"ok: sequence {loaded.sequence}, channel {loaded.channel}, signed by {loaded.key_id}, "
              f"{len(loaded.engine)} engine packages, control {len(loaded.control)}, app {'yes' if loaded.app else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
