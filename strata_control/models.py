"""Bounded, read-only discovery of setup-written local model run configs.

The setup installer writes strata-*.json beside setup.py. A configured model is
not automatically a compatible or loadable model: component presence and native
engine compatibility are reported separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from strata_control.doctor import inspect as inspect_components
from strata_control.gguf_preflight import inspect_metadata
from strata_control.profiles import ProfileError


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}\Z")
MAX_CONFIG_BYTES = 1024 * 1024
MAX_CONFIGS = 128
MAX_IMPORT_SHARDS = 128
SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf\Z", re.I)
SUPPORTED_SELECTIONS = frozenset({"qwen/Q2_0", "qwen/IQ2_XS", "qwen/IQ3_XXS", "qwen/IQ3_S",
                                   "swift/IQ2_XS", "swift/IQ3_XXS", "coder/IQ1_M",
                                   "unsloth/UD-IQ4_XS", "unsloth/UD-Q4_K_XL"})


def install_dir(override: str | None = None) -> Path:
    raw = override or os.environ.get("STRATA_INSTALL_DIR")
    if raw is None:
        return ROOT
    path = Path(raw).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ProfileError("install directory must be an existing absolute directory", 5)
    return path.resolve()


def _model_id(path: Path) -> str | None:
    if not path.name.startswith("strata-") or not path.name.endswith(".json") or \
            path.name.endswith(".shared-settings.json"):
        return None
    value = path.name[len("strata-"):-len(".json")]
    return value if MODEL_ID.fullmatch(value) else None


def _paths(directory: Path) -> list[tuple[str, Path]]:
    try:
        entries = [(name, path) for path in directory.iterdir() if (name := _model_id(path)) is not None]
    except OSError:
        raise ProfileError("cannot read installed model directory", 5) from None
    if len(entries) > MAX_CONFIGS:
        raise ProfileError("more than 128 model configs need inspection", 5)
    return sorted(entries, key=lambda item: item[0])


def _read(path: Path) -> tuple[bytes, dict]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise ProfileError("run config must not be a symlink", 5)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            details = os.fstat(stream.fileno())
            if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_CONFIG_BYTES:
                raise ProfileError("run config must be a regular file under 1 MiB", 5)
            raw = stream.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            raise ProfileError("run config exceeds 1 MiB", 5)
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ProfileError):
            raise
        raise ProfileError("run config is unreadable or invalid JSON", 5) from None
    if not isinstance(value, dict) or not isinstance(value.get("exe"), str) or not value["exe"] or \
            not isinstance(value.get("tokenizer"), str) or not value["tokenizer"] or \
            not isinstance(value.get("args"), list) or \
            not all(isinstance(arg, str) for arg in value["args"]):
        raise ProfileError("run config lacks engine, tokenizer or string arguments", 5)
    return raw, value


def _short(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return "".join(char if char.isprintable() and char not in "\x1b\x7f" else " " for char in value)[:120]


def _arg(args: list[str], flag: str) -> str | None:
    try:
        index = args.index(flag)
        return args[index + 1] if index + 1 < len(args) else None
    except ValueError:
        return None


def _summary(model_id: str, path: Path, raw: bytes, cfg: dict, report: dict) -> dict:
    context = _arg(cfg["args"], "--max-context")
    context_tokens = int(context) if isinstance(context, str) and 1 <= len(context) <= 8 and \
        context.isascii() and context.isdecimal() and 1 <= int(context) <= 16_777_216 else None
    backend = cfg.get("backend")
    return {"id": model_id, "name": _short(cfg.get("model_name")) or model_id,
            "source_config": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "backend": _short(backend) if backend is not None else None,
            "context_tokens": context_tokens,
            "component_status": "present" if report["ok"] else "missing_or_invalid",
            "issue_codes": [item.get("code") for item in report["issues"]][:32],
            "execution_compatibility": "unverified"}


def _inspect_one(model_id: str, path: Path) -> tuple[dict, dict]:
    raw, cfg = _read(path)
    try:
        report = inspect_components(path)
        after, _ = _read(path)
    except (OSError, ValueError):
        raise ProfileError("run config changed or became invalid during inspection", 6) from None
    if after != raw:
        raise ProfileError("run config changed during inspection; retry", 6)
    return _summary(model_id, path, raw, cfg, report), report


def list_installed(override: str | None = None) -> dict:
    directory = install_dir(override)
    models = []
    for model_id, path in _paths(directory):
        try:
            summary, _ = _inspect_one(model_id, path)
        except ProfileError as exc:
            summary = {"id": model_id, "source_config": str(path), "component_status": "invalid",
                       "error": str(exc), "execution_compatibility": "unverified"}
        models.append(summary)
    return {"scope": "local_run_configs", "install_dir": str(directory), "models": models,
            "model_count": len(models), "discovery_limit": MAX_CONFIGS}


def inspect(model_id: str, override: str | None = None) -> dict:
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
        raise ProfileError("model ID must be a setup config name without a path")
    directory = install_dir(override)
    choices = dict(_paths(directory))
    path = choices.get(model_id)
    if path is None:
        raise ProfileError(f"model {model_id!r} is not configured in this installation", 5)
    summary, report = _inspect_one(model_id, path)
    return {"scope": "local_run_config", "install_dir": str(directory), "model": summary,
            "components": report}


def _gguf_header(path: Path) -> dict:
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise ProfileError("GGUF shard is a symlink", 5)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            details = os.fstat(stream.fileno())
            if not stat.S_ISREG(details.st_mode) or details.st_size < 8:
                raise ProfileError("GGUF shard is not a nonempty regular file", 5)
            header = stream.read(8)
    except OSError:
        raise ProfileError("GGUF shard is unreadable", 5) from None
    if header[:4] != b"GGUF":
        raise ProfileError("file does not begin with a GGUF header", 5)
    version = int.from_bytes(header[4:8], "little")
    if version not in (2, 3):
        raise ProfileError("GGUF header version is not recognized", 5)
    return {"name": path.name, "path": str(path), "bytes": details.st_size,
            "gguf_version": version}


def _candidate_selection(shards: list[Path]) -> dict | None:
    fingerprint_file = ROOT / "data" / "gguf_fingerprints.json"
    try:
        if fingerprint_file.stat().st_size > 128 * 1024:
            raise ProfileError("model fingerprint catalog exceeds 128 KiB", 5)
        catalog = json.loads(fingerprint_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise ProfileError("model fingerprint catalog is unavailable", 5) from None
    names = {item.name for item in shards}
    for selection in sorted(SUPPORTED_SELECTIONS):
        row = catalog.get(selection) if isinstance(catalog, dict) else None
        expected = row.get("files") if isinstance(row, dict) else None
        if isinstance(expected, list) and expected and names == {Path(name).name for name in expected}:
            family, model = selection.split("/", 1)
            expected_experts = row.get("expert_bytes")
            return {"family": family, "model": model, "basis": "known_shard_names_only",
                    "expected_expert_types": sorted(expected_experts) if isinstance(expected_experts, dict) else [],
                    "expected_tensor_count": row.get("tensors") if type(row.get("tensors")) is int else None,
                    "setup_args_hint": ["--family", family, "--model", model,
                                        "--gguf-dir", str(shards[0].parent)]}
    return None


def plan_import(raw_path: str) -> dict:
    """Inspect a local candidate without copying, parsing tensors or adopting it."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ProfileError("model import path must be absolute")
    try:
        details = path.lstat()
    except OSError:
        raise ProfileError("model import path does not exist", 5) from None
    if stat.S_ISLNK(details.st_mode):
        raise ProfileError("model import path must not be a symlink", 5)
    if stat.S_ISDIR(details.st_mode):
        try:
            shards = []
            ignored_auxiliary = []
            for child in path.iterdir():
                if child.name.lower().endswith(".gguf"):
                    if child.name.lower().startswith("mmproj"):
                        ignored_auxiliary.append(child.name)
                        continue
                    shards.append(child)
                    if len(shards) > MAX_IMPORT_SHARDS:
                        raise ProfileError("more than 128 GGUF candidates in the directory", 5)
        except OSError:
            raise ProfileError("cannot inspect GGUF directory", 5) from None
    elif stat.S_ISREG(details.st_mode) and path.name.lower().endswith(".gguf"):
        ignored_auxiliary = []
        match = SHARD.fullmatch(path.name)
        if match:
            total = int(match.group(3))
            if not 1 <= total <= MAX_IMPORT_SHARDS:
                raise ProfileError("GGUF shard count must be in 1..128", 5)
            shards = [path.with_name(f"{match.group(1)}-{i:05d}-of-{total:05d}.gguf")
                      for i in range(1, total + 1)]
        else:
            shards = [path]
    else:
        raise ProfileError("model import needs a GGUF file or directory", 5)
    if not shards:
        raise ProfileError("no GGUF files found at the model import path", 5)
    shards.sort(key=lambda item: item.name)
    groups = {}
    for item in shards:
        match = SHARD.fullmatch(item.name)
        if match:
            key = (match.group(1), int(match.group(3)))
            groups.setdefault(key, set()).add(int(match.group(2)))
        else:
            groups.setdefault((item.name, 1), set()).add(1)
    issues = []
    if len(groups) != 1:
        issues.append("multiple_model_groups")
    if len(groups) == 1:
        (_, total), indices = next(iter(groups.items()))
        if total > MAX_IMPORT_SHARDS or indices != set(range(1, total + 1)):
            issues.append("incomplete_or_invalid_shard_set")
    files = []
    for item in shards:
        try:
            files.append(_gguf_header(item))
        except ProfileError as exc:
            issues.append(f"{item.name}: {exc}")
    metadata = None
    observed_experts: set[str] = set()
    observed_tensor_count = 0
    if files and not issues:
        for item in shards:
            try:
                inspected = inspect_metadata(item)
            except ProfileError as exc:
                issues.append(f"{item.name}: {exc}")
                continue
            if metadata is None:
                metadata = inspected
            observed_experts.update(inspected["expert_tensor_types"])
            observed_tensor_count += inspected["tensor_count_scanned"]
            if inspected["architecture"] is not None and inspected["architecture"] != "qwen4exp":
                issues.append(f"{item.name}: unsupported_architecture")
    if metadata is not None:
        if metadata["architecture"] != "qwen4exp":
            issues.append("unsupported_or_missing_architecture")
        if metadata["tokenizer_model"] != "gpt2" or not metadata["tokenizer_tokens_present"] or \
                not metadata["tokenizer_merges_present"]:
            issues.append("unsupported_or_incomplete_tokenizer_metadata")
    candidate = _candidate_selection(shards)
    if candidate is None:
        issues.append("unrecognized_or_unsupported_model_variant")
    elif metadata is not None:
        expected = set(candidate["expected_expert_types"])
        if not observed_experts or observed_experts - expected:
            issues.append("missing_or_unexpected_expert_quantization")
    return {"source": str(path), "scope": "local_gguf_candidate", "files": files,
            "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files),
            "metadata": metadata, "candidate_selection": candidate,
            "observed_expert_types": sorted(observed_experts),
            "observed_tensor_count": observed_tensor_count,
            "ignored_auxiliary_files": sorted(ignored_auxiliary),
            "file_set_status": "plausible" if not issues else "invalid_or_incomplete",
            "issues": issues, "adoption_status": "requires_architecture_quantization_tokenizer_and_engine_checks",
            "execution_compatibility": "unverified", "would_write": False,
            "notes": ["A known shard name is only a selection hint; it does not prove provenance or file integrity",
                      "Tensor directory, model-pack, engine version and hardware checks remain before adoption",
                      "No model is adopted or copied by this preview"]}
