#!/usr/bin/env bash
# Write a tester's README with the checksums and sizes of the files it names, computed from the files themselves.
#
#   apps/desktop/scripts/make-tester-readme.sh <template> <out.txt> --appimage <file name> --files-dir <dir> [--files-dir <dir> ...]
#
# Why: a README that carries a hand-pasted sha256 goes stale the moment the file is rebuilt (the Ubuntu 26.04 tester compared the file
# with the old build's hash). The template holds placeholders instead, and this script fills them from the real files:
#   @APPIMAGE@            the AppImage's file name
#   @SHA256:<name>@       sha256 of the file <name> found in one of the --files-dir folders
#   @SIZE_GB:<name>@      its size in GB with one decimal
#   @INCLUDE:<file>@      (alone on a line) the contents of <file> from the template's folder, with the same placeholders
# It refuses to write the README when a named file is missing or a placeholder is left over, so a stale or empty hash cannot ship.
# The same check is done by `--verify <readme> --files-dir <dir>`: every "<64 hex>  <name>" or "sha256 <64 hex>" pair that names a file
# present in the folders must match that file.
set -euo pipefail

usage() { echo "usage: $0 <template> <out.txt> --appimage <name> --files-dir <dir> [--files-dir <dir> ...]" >&2; echo "       $0 --verify <readme> --files-dir <dir> [...]" >&2; exit 2; }

VERIFY=""; TEMPLATE=""; OUT=""; APPIMAGE=""; DIRS=()
if [ "${1:-}" = "--verify" ]; then VERIFY="${2:?readme}"; shift 2; else TEMPLATE="${1:?template}"; OUT="${2:?out}"; shift 2 || usage; fi
while [ $# -gt 0 ]; do
  case "$1" in
    --appimage) APPIMAGE="$2"; shift 2 ;;
    --files-dir) DIRS+=("$2"); shift 2 ;;
    *) usage ;;
  esac
done
[ ${#DIRS[@]} -gt 0 ] || usage

find_file() { local d; for d in "${DIRS[@]}"; do [ -f "$d/$1" ] && { echo "$d/$1"; return 0; }; done; return 1; }
sha_of() { sha256sum "$1" | awk '{print $1}'; }
size_gb() { awk -v b="$(stat -c %s "$1" 2>/dev/null || stat -f %z "$1")" 'BEGIN{printf "%.1f", b/1000000000}'; }

if [ -n "$VERIFY" ]; then
  bad=0; seen=0
  while IFS= read -r line; do
    # "<hash>  <name>" or "<name>   <hash>" or "sha256 <hash>"
    hash="$(grep -Eo '[0-9a-f]{64}' <<<"$line" | head -1 || true)"
    [ -n "$hash" ] || continue
    name="$(sed -E 's/[0-9a-f]{64}//; s/^[[:space:]]*(sha256|Checks:)?[[:space:]]*//; s/[[:space:]]*$//' <<<"$line")"
    f="$(find_file "$name" 2>/dev/null || true)"
    [ -n "$f" ] || continue
    seen=$((seen + 1))
    if [ "$(sha_of "$f")" != "$hash" ]; then echo "STALE: $name is $(sha_of "$f"), the README says $hash" >&2; bad=$((bad + 1)); fi
  done < "$VERIFY"
  echo "$seen checksum line(s) checked, $bad stale"
  [ "$bad" = 0 ] && [ "$seen" -gt 0 ]
  exit
fi

[ -n "$APPIMAGE" ] || usage
TDIR="$(cd "$(dirname "$TEMPLATE")" && pwd)"
TMP="$(mktemp)"; trap 'rm -f "$TMP" "$TMP.2"' EXIT
# includes first (one level), then the placeholders
while IFS= read -r line || [ -n "$line" ]; do
  if [[ "$line" =~ ^@INCLUDE:([^@]+)@$ ]]; then cat "$TDIR/${BASH_REMATCH[1]}"; else printf '%s\n' "$line"; fi
done < "$TEMPLATE" | sed "s|@APPIMAGE@|$APPIMAGE|g" > "$TMP"

# @SHA256:name@ and @SIZE_GB:name@
while IFS= read -r token; do
  kind="${token#@}"; kind="${kind%%:*}"; name="${token#@*:}"; name="${name%@}"
  f="$(find_file "$name")" || { echo "make-tester-readme: no file named $name in: ${DIRS[*]} (a README must not carry a hash it did not compute)" >&2; exit 1; }
  case "$kind" in
    SHA256) value="$(sha_of "$f")" ;;
    SIZE_GB) value="$(size_gb "$f")" ;;
    *) echo "make-tester-readme: unknown placeholder $token" >&2; exit 1 ;;
  esac
  # replace every occurrence of this exact token (the token contains no sed-special characters except . and -)
  esc="$(printf '%s' "$token" | sed 's/[.[\*^$/]/\&/g')"
  sed "s/$esc/$value/g" "$TMP" > "$TMP.2" && mv "$TMP.2" "$TMP"
done < <(grep -Eo '@(SHA256|SIZE_GB):[^@ ]+@' "$TMP" | sort -u)

if grep -Eq '@[A-Z0-9_]+[:@]' "$TMP"; then echo "make-tester-readme: placeholders left over:" >&2; grep -En '@[A-Z0-9_]+[:@]' "$TMP" >&2; exit 1; fi
cp "$TMP" "$OUT"
echo "wrote $OUT"
